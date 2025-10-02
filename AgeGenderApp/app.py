from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Detection
import os
import cv2
import base64
import numpy as np
from datetime import datetime
from detect_utils import detect_age_gender_image, detect_age_gender_video, detect_age_gender_frame

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'your-secret-key'

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables immediately after app and db initialized
with app.app_context():
    db.create_all()

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for('index'))
    file = request.files["file"]
    ext = file.filename.rsplit('.', 1)[1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output_" + file.filename)
        results, processed_path = detect_age_gender_image(filepath, output_path)
        log_detections(results, "image")
        return render_template("index.html", results=results, user_image=os.path.basename(processed_path))
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        output_vid_path = os.path.join(app.config['UPLOAD_FOLDER'], "output_" + file.filename)
        results, video_path = detect_age_gender_video(filepath, output_vid_path)
        flat_results = [label for frame in results for label in frame if label]
        log_detections(flat_results, "video")
        from collections import Counter
        summary = Counter(flat_results)
        return render_template("index.html", results=summary, user_video=os.path.basename(video_path))
    else:
        flash("File type not supported.", "warning")
        return redirect(url_for('index'))

def log_detections(labels, source):
    for result in labels:
        age, gender = None, None
        if ',' in result:
            parts = result.split(',')
            gender = parts[0].strip()
            age = parts[1].strip()
        det = Detection(timestamp=datetime.now(), age=age, gender=gender, source=source, user_id=current_user.id)
        db.session.add(det)
    db.session.commit()

@app.route("/live_detect", methods=["POST"])
@login_required
def live_detect():
    data = request.json
    img_data = data.get("image")
    if not img_data:
        return jsonify({"error": "No image data"}), 400

    img_bytes = base64.b64decode(img_data.split(",")[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results, processed_frame = detect_age_gender_frame(frame)
    log_detections(results, "live")
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    return jsonify({"results": results, "image": processed_b64})

@app.route("/dashboard")
@login_required
def dashboard():
    stats = Detection.query.filter_by(user_id=current_user.id).all()
    # You can process stats here for charts
    return render_template("dashboard.html", detections=stats)

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("index"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "warning")
            return redirect(url_for("signup"))
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash("Signup successful. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
