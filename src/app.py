import os
import time
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape

import frontmatter
import markdown
import yaml
from bs4 import BeautifulSoup
from flask import Flask, Response, redirect, render_template, request, send_from_directory, url_for
from slugify import slugify

from src.text_analytics import compute_tfidf, cosine_similarity_manual, tokenize

DEV = os.getenv("DEV", False) == "True"

# Per-process article cache: {domain: (timestamp, articles_list)}
_articles_cache: dict = {}
_ARTICLES_CACHE_TTL = 300  # seconds


def _normalize_date(date_val):
  """Return (ISO string, display string) from a raw YAML date value (str or date object)."""
  if not date_val:
    return "", ""
  if hasattr(date_val, "strftime"):
    dt = datetime(date_val.year, date_val.month, date_val.day) if not isinstance(date_val, datetime) else date_val
  else:
    try:
      dt = datetime.strptime(str(date_val), "%Y-%m-%d")
    except ValueError:
      return str(date_val), str(date_val)
  return dt.strftime("%Y-%m-%d"), dt.strftime("%B %d, %Y")


def _render_body(text):
  """Render Markdown to HTML and add rel='nofollow noopener' + target='_blank' to external links."""
  html = markdown.markdown(text, extensions=["extra", "codehilite", "toc"])
  soup = BeautifulSoup(html, "html.parser")
  for tag in soup.find_all("a", href=True):
    href = tag["href"]
    if href.startswith("http://") or href.startswith("https://"):
      tag["rel"] = ["nofollow", "noopener"]
      if not tag.get("target"):
        tag["target"] = "_blank"
  return str(soup)


# Add this near the top of your file, with other imports
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}


def prefered_domain(request):
  """Return the preferred domain based on the request host."""
  domain = request.host if not DEV else "voltapowers.com"

  if domain.endswith(".fly.dev"):
    domain = "voltapowers.com"

  return domain


def find_similar_image_filename(requested_filename, directory="content/assets", threshold=0.9):
  """
    Find an image in the directory with a filename most similar to the requested filename
    using cosine similarity on tokenized filenames (without extensions).
    """
  requested_name, _ = os.path.splitext(requested_filename.lower())
  image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS]

  if not image_files:
    return None

  image_names = [os.path.splitext(f.lower())[0] for f in image_files]

  # Compute TF-IDF for image names to get vocabulary
  tfidf_vectors, vocab = compute_tfidf(image_names)
  if not tfidf_vectors:
    return None

  # Compute TF-IDF for requested name using the same vocabulary
  query_vector = compute_tfidf([requested_name], vocab=vocab)[0][0]

  # Compute cosine similarities
  sim_scores = [(image_files[i], cosine_similarity_manual(query_vector, tfidf_vectors[i])) for i in range(len(image_files))]

  # Sort by similarity score (descending)
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  # Return the top match if its similarity is above the threshold
  if sim_scores and sim_scores[0][1] > threshold:
    print(f"Found similar image: {sim_scores[0][0]} for {requested_filename} with similarity {sim_scores[0][1]}")
    return sim_scores[0][0]

  return None


def _theme(config):
  """Return the theme name from config, defaulting to 'news'."""
  return config.get("site", {}).get("theme", "news")


def load_domain_config():
  """Load domain config based on request host. Don’t mess this up, or we’re serving 404s all day."""

  domain = prefered_domain(request)
  # if domain is not in list of domains/* then set default config is os ls to know
  if not os.path.exists(f"domains/{domain}.yml") and not os.path.exists(f"domains/{domain}.yaml"):
    print(f"Config for {domain} not found. Falling back to default.")
    domain = "voltapowers.com"

  config_paths = [f"domains/{domain}.yml", f"domains/{domain}.yaml"]
  config_path = next((path for path in config_paths if os.path.exists(path)), None)
  if not os.path.exists(config_path):
    print(f"Config for {domain} not found at {config_path}. Falling back to default.")
    return {
      "site": {"name": "Default Blog", "description": "A blog platform", "keywords": []},
      "content": {"topics": [], "articles_path": f"content/{domain}/articles", "default_image": "/content/assets/tech-bg.jpg"},
      "seo": {"og_image": "/content/assets/tech-bg.jpg", "twitter_image": "/content/assets/tech-bg.jpg"},
    }
  try:
    with open(config_path, "r") as f:
      return yaml.safe_load(f)
  except yaml.YAMLError as e:
    print(f"Invalid YAML in {config_path}: {str(e)}. Default config it is, then.")
    return {
      "site": {"name": "Default Blog", "description": "A blog platform", "keywords": []},
      "content": {"topics": [], "articles_path": f"content/{domain}/articles", "default_image": "/content/assets/tech-bg.jpg"},
      "seo": {"og_image": "/content/assets/tech-bg.jpg", "twitter_image": "/content/assets/tech-bg.jpg"},
    }


def load_articles(domain):
  """Load articles from content/[domain]/articles with a short TTL in-process cache."""
  now = time.time()
  cached = _articles_cache.get(domain)
  if cached and (now - cached[0]) < _ARTICLES_CACHE_TTL:
    return cached[1]

  articles_path = f"content/{domain}/articles"
  articles = []
  if not os.path.exists(articles_path):
    print(f"No articles directory for {domain} at {articles_path}. You got nothing to show.")
    return []
  for filename in os.listdir(articles_path):
    if filename.endswith(".md"):
      try:
        with open(os.path.join(articles_path, filename), "r") as f:
          post = frontmatter.load(f)
          raw_date = post.get("date", "")
          date_iso, date_display = _normalize_date(raw_date)
          words = len(post.content.split())
          articles.append(
            {
              "title": post.get("title", "Untitled"),
              "slug": post.get("slug", filename[:-3]),
              "meta_description": post.get("meta_description", ""),
              "meta_keywords": post.get("meta_keywords", []),
              "og_title": post.get("title", "Untitled"),
              "og_description": post.get("og_description", ""),
              "og_image": post.get("og_image", ""),
              "author": post.get("author", "Anonymous"),
              "date": date_iso,          # raw ISO kept for sorting
              "date_iso": date_iso,      # ISO format for JSON-LD / OG tags
              "date_display": date_display,
              "body": post.content,
              "body_html": _render_body(post.content),
              "topic": post.get("topic", ""),
              "original_url": post.get("original_url", ""),
              "reading_time": max(1, round(words / 200)),
            }
          )
      except Exception as e:
        print(f"Error loading article {filename}: {str(e)}. Skipping.")

  sorted_list = sorted(articles, key=lambda x: x["date"] or "1970-01-01", reverse=True)
  for article in sorted_list:
    article["date"] = article.pop("date_display")

  _articles_cache[domain] = (now, sorted_list)
  return sorted_list


def get_related_articles(articles, current_article):
  """Find related articles using TF-IDF with a lower threshold."""
  if not articles or not current_article:
    return articles[:10] if articles else []
  documents = [f"{a['title']} {a['meta_description']} {' '.join(a['meta_keywords'])}" for a in articles]
  tfidf_vectors, vocab = compute_tfidf(documents)
  current_text = f"{current_article['title']} {current_article['meta_description']} {' '.join(current_article['meta_keywords'])}"
  current_vector = compute_tfidf([current_text], vocab=vocab)[0][0]

  similarities = [(articles[i], cosine_similarity_manual(current_vector, vector)) for i, vector in enumerate(tfidf_vectors)]
  similarities.sort(key=lambda x: x[1], reverse=True)
  # Lowered threshold to 0.1 for more matches
  return [article for article, score in similarities[:10] if article["slug"] != current_article["slug"]]


def create_app():
  app = Flask(__name__, template_folder="../templates", static_folder="../static")
  app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 31536000  # 1-year cache for static assets

  @app.context_processor
  def inject_config():
    """Inject domain config into all templates. Don’t ask me to do this manually."""
    # if not dev prefer https
    if not DEV:
      app.config["PREFERRED_URL_SCHEME"] = "https"
    cfg = load_domain_config()
    return dict(config=cfg, theme=_theme(cfg))

  @app.route("/")
  def index():
    """Homepage with featured and recent articles. No articles? Enjoy the silence."""
    domain = prefered_domain(request)
    articles = load_articles(domain)
    latest_article = articles[0] if articles else None
    latest_articles = articles[1:5] if len(articles) > 1 else []
    related_articles = get_related_articles(articles, None)
    return render_template(f"themes/{_theme(load_domain_config())}/index.html", latest_article=latest_article, latest_articles=latest_articles, related_articles=related_articles)

  @app.route("/article/<slug>")
  def article(slug):
    """Serve an article by slug. Wrong slug? You’re getting a sneaky 404."""
    domain = prefered_domain(request)
    articles = load_articles(domain)
    article = next((a for a in articles if a["slug"] == slug), None)
    if not article:
      return handle_404(None)
    related_articles = get_related_articles(articles, article)
    return render_template(f"themes/{_theme(load_domain_config())}/article.html", article=article, related_articles=related_articles)

  @app.route("/category/<category>")
  def category(category):
    """Serve articles by category (keyword)."""
    domain = prefered_domain(request)
    articles = load_articles(domain)
    matching_articles = [a for a in articles if category in [slugify(kw) for kw in a["meta_keywords"]]]
    return render_template(
      "index.html",
      latest_article=None,
      latest_articles=matching_articles[:10],  # Limit to 10 articles
      related_articles=articles[:10],
      category_name=category.replace("-", " ").title(),
      category_slug=category,
    )

  @app.route("/search")
  def search():
    """Search articles with TF-IDF."""
    query = request.args.get("q", "").lower()
    if not query:
      return redirect(url_for("index"))
    domain = prefered_domain(request)
    articles = load_articles(domain)
    results = []
    if articles:
      documents = [f"{a['title']} {a['meta_description']} {' '.join(a['meta_keywords'])} {a['body']}" for a in articles]
      tfidf_vectors, vocab = compute_tfidf(documents)
      query_vector = compute_tfidf([query], vocab=vocab)[0][0]
      for i, article in enumerate(articles):
        score = cosine_similarity_manual(query_vector, tfidf_vectors[i])
        if score > 0.1:
          results.append(article)
    results.sort(key=lambda x: x["date"] or "1970-01-01", reverse=True)
    return render_template(f"themes/{_theme(load_domain_config())}/index.html", latest_article=None, latest_articles=results, related_articles=articles[:3], search_query=query)

  @app.route("/contact", methods=["GET", "POST"])
  def contact():
    """Contact form."""
    if request.method == "POST":
      name = request.form.get("name")
      email = request.form.get("email")
      message = request.form.get("message")
      if name and email and message:
        print(f"Contact form submission: {name}, {email}, {message}")
        return redirect(url_for("thank_you"))
      print(f"Invalid contact form submission: {name}, {email}, {message}")
    return render_template(f"themes/{_theme(load_domain_config())}/contact.html")

  @app.route("/subscribe", methods=["POST"])
  def subscribe():
    """Newsletter signup."""
    email = request.form.get("email")
    if email:
      print(f"Newsletter subscription: {email}")
      return redirect(url_for("thank_you"))
    print("Invalid subscription attempt: No email provided.")
    return redirect(url_for("index"))

  @app.route("/thank_you")
  def thank_you():
    """Thank you page for forms."""
    return render_template(f"themes/{_theme(load_domain_config())}/thank_you.html")

  @app.route("/guest_poster")
  def guest_poster():
    """Guest poster page."""
    return render_template(f"themes/{_theme(load_domain_config())}/guest_poster.html")

  @app.route("/privacy")
  def privacy():
    """Privacy policy."""
    return render_template(f"themes/{_theme(load_domain_config())}/privacy.html")

  @app.route("/sitemap.xml")
  def sitemap():
    """Sitemap for SEO."""
    domain = prefered_domain(request)
    articles = load_articles(domain)
    base_url = url_for("index", _external=True, _scheme="https").rstrip("/")
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">\n'
    xml += f'<url><loc>{url_for("index", _external=True, _scheme="https")}</loc><changefreq>daily</changefreq><priority>1.0</priority></url>\n'
    for article in articles:
      lastmod = article.get("date_iso", "")
      xml += f'<url><loc>{url_for("article", slug=article["slug"], _external=True, _scheme="https")}</loc>'
      if lastmod:
        xml += f"<lastmod>{lastmod}</lastmod>"
      xml += "<changefreq>weekly</changefreq><priority>0.8</priority>"
      og_image = article.get("og_image", "")
      if og_image:
        image_url = og_image if og_image.startswith("http") else f"{base_url}{og_image}"
        xml += f"<image:image><image:loc>{xml_escape(image_url)}</image:loc><image:title>{xml_escape(article.get('title', ''))}</image:title></image:image>"
      xml += "</url>\n"
    categories = set()
    for article in articles:
      for kw in article.get("meta_keywords", []):
        categories.add(slugify(kw))
    for cat in sorted(categories):
      xml += f'<url><loc>{url_for("category", category=cat, _external=True, _scheme="https")}</loc><changefreq>weekly</changefreq><priority>0.6</priority></url>\n'
    xml += "</urlset>"
    return Response(xml, mimetype="application/xml")

  @app.route("/robots.txt")
  def robots_txt():
    """Robots.txt for SEO crawl control."""
    sitemap_url = url_for("sitemap", _external=True, _scheme="https")
    content = f"User-agent: *\nAllow: /\nSitemap: {sitemap_url}\n"
    return Response(content, mimetype="text/plain")

  @app.route("/content/assets/<path:filename>")
  def serve_content_images(filename):
    """Serve images from /content/assets with fallback."""
    _, ext = os.path.splitext(filename.lower())
    is_image = ext in IMAGE_EXTENSIONS
    content_relative_dir_path = "../content/assets"
    if is_image:
      file_path = os.path.join("content/assets", filename)
      if os.path.exists(file_path) and os.path.isfile(file_path):
        print(f"Serving image: {file_path}")
        return send_from_directory(content_relative_dir_path, filename)
      else:
        print(f"Image not found: {file_path}")
        similar_image = find_similar_image_filename(filename)
        if similar_image:
          similar_image_path = os.path.join("content/assets", similar_image)
          if os.path.exists(similar_image_path):
            print(f"Serving similar image: {similar_image}")
            return send_from_directory(content_relative_dir_path, similar_image)
        default_image = "not-found.jpg"
        default_image_path = os.path.join("content/assets", default_image)
        if os.path.exists(default_image_path):
          print(f"Serving fallback image: {default_image}")
          return send_from_directory(content_relative_dir_path, default_image)
        return send_from_directory(content_relative_dir_path, "vancouver-bg.jpg")
    return render_template(f"themes/{_theme(load_domain_config())}/404.html", related_articles=[]), 404

  @app.template_filter("markdown")
  def markdown_filter(text):
    return _render_body(text)

  @app.template_filter("slugify")
  def slugify_filter(text):
    """Convert text to a URL-friendly slug."""
    return slugify(text)

  @app.errorhandler(404)
  def handle_404(e):
    """Handle 404 with article suggestions."""
    domain = prefered_domain(request)
    articles = load_articles(domain)
    path = request.path.strip("/")
    slug = path.split("/")[-1] if path.startswith("article/") else path
    for article in articles:
      if slug.lower() in article["slug"].lower() or slug.lower() in article["title"].lower():
        return render_template(f"themes/{_theme(load_domain_config())}/article.html", article=article, related_articles=get_related_articles(articles, article)), 200
    if articles:
      documents = [f"{a['title']} {a['meta_description']} {' '.join(a['meta_keywords'])}" for a in articles]
      tfidf_vectors, vocab = compute_tfidf(documents)
      slug_vector = compute_tfidf([slug], vocab=vocab)[0][0]
      similarities = [(articles[i], cosine_similarity_manual(slug_vector, vector)) for i, vector in enumerate(tfidf_vectors)]
      similarities.sort(key=lambda x: x[1], reverse=True)
      if similarities and similarities[0][1] > 0.7:
        matched_article = similarities[0][0]
        return render_template(f"themes/{_theme(load_domain_config())}/article.html", article=matched_article, related_articles=get_related_articles(articles, matched_article)), 200
    return render_template(f"themes/{_theme(load_domain_config())}/404.html", related_articles=articles[:10]), 200

  @app.route("/healthz")
  def health_check():
    """Health check endpoint for load balancers."""
    return "OK", 200

  @app.before_request
  def handle_fly_domain_redirect():
    domain = prefered_domain(request)
    if request.host.endswith(".fly.dev"):
      canonical_url = f"https://{domain}{request.path}"
      if request.query_string:
        canonical_url += f"?{request.query_string.decode('utf-8')}"
      return redirect(canonical_url, code=301)
    request.canonical_url = f"https://{domain}{request.path}"

  return app
