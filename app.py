from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    Response,
)
from flask_sqlalchemy import SQLAlchemy
import random
import os
import csv
import json
from io import StringIO
from sqlalchemy import func
from statistics import mean, stdev
from functools import wraps
import numpy as np

# Add your admin credentials here
ADMIN_USERNAME = "jhkim"
ADMIN_PASSWORD = "aaai2025"
PROJECT_NAME = "MOS"


app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{PROJECT_NAME}.db"
db = SQLAlchemy(app)

# Function to load samples for a given task type (emotion or speaker)
def load_samples_for_task(task_type):
    """Load samples based on task type (emotion or speaker)"""
    samples = []
    
    # Determine the base path based on task type
    if "emotion" in task_type.lower():
        base_path = os.path.join("static", "samples", "MOS4", "emotion")
    elif "speaker" in task_type.lower():
        base_path = os.path.join("static", "samples", "MOS4", "speaker")
    else:
        print(f"Unknown task type: {task_type}")
        return samples
    
    print(f"Loading samples from: {base_path}")
    print(f"Directory exists: {os.path.exists(base_path)}")
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        # Check if this directory contains video files (i.e., it's a leaf directory with models)
        video_files = [f for f in files if f.endswith('.mp4')]
        
        if video_files and 'gt.mp4' in video_files:
            # Get the relative path from base_path for the sample name
            rel_path = os.path.relpath(root, base_path)
            
            # Extract model files
            file_list = []
            for video_file in sorted(video_files):
                model_name = os.path.splitext(video_file)[0]  # e.g., 'gt', 'proposed', etc.
                file_path = os.path.join(root, video_file)
                
                file_list.append({
                    "model_name": model_name,
                    "file_name": video_file,
                    "file_path": file_path.replace("static/", ""),
                    "text": rel_path,  # Use relative path as description
                    "media_type": "video",
                    "mime_type": "video/mp4"
                })
            
            if file_list:
                samples.append({
                    "file_name": rel_path,
                    "files": file_list
                })
    
    print(f"Found {len(samples)} samples for task type: {task_type}")
    return samples


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    task_type = db.Column(db.String(50), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    sample_sequence = db.Column(db.Text, nullable=True)  # Store the sequence of samples
    num_pages = db.Column(db.Integer, nullable=True)  # Store the number of pages
    scores = db.relationship("Score", backref="user", lazy=True)


class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    file_name = db.Column(db.String(100), nullable=False)
    score = db.Column(db.Integer, nullable=False)


# Ensure the database is created within the application context
with app.app_context():
    db.create_all()


# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("results"))
        else:
            error = "Invalid Credentials. Please try again."
            return render_template("login.html", error=error)

    return render_template("login.html")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" not in session or not session["logged_in"]:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/reset_db")
@login_required
def reset_db():
    db.drop_all()  # Drop all tables
    db.create_all()  # Recreate all tables
    return redirect(url_for("start"))


@app.route("/", methods=["GET", "POST"])
def start():
    if request.method == "POST":
        name = request.form["name"]
        task_type = request.form["task_type"]
        user = User.query.filter_by(name=name).first()

        if not user:
            # New user, create a new record
            # Load samples for the selected task type
            samples = load_samples_for_task(task_type)
            
            if not samples:
                flash(f"No samples found for task type: {task_type}. Please check the sample directory.")
                return redirect(url_for("start"))
            
            # Select a subset of samples (adjust num_pages as needed)
            num_pages = min(20, len(samples))  # Use up to 20 samples or all available
            selected_samples = random.sample(samples, num_pages) if len(samples) > num_pages else samples
            sample_sequence = json.dumps(selected_samples)  # Store the sequence as a JSON string

            user = User(
                name=name,
                task_type=task_type,
                num_pages=num_pages,
                sample_sequence=sample_sequence,
            )
            db.session.add(user)
            db.session.commit()
        elif user.completed:
            flash("You have already completed the survey.", "info")
            return redirect(url_for("end"))

        session["user_id"] = user.id
        session["page"] = 1
        session["task_type"] = user.task_type

        return redirect(url_for("instructions"))

    return render_template("start.html")


@app.route("/instructions")
def instructions():
    if "user_id" not in session:
        return redirect(url_for("start"))

    return render_template("instructions.html", task_type=session["task_type"])


@app.route("/score", methods=["GET", "POST"])
def score():
    if "user_id" not in session:
        return redirect(url_for("start"))

    user_id = session["user_id"]
    user = db.session.get(User, user_id)
    page = session.get("page", 1)
    sample_sequence = json.loads(user.sample_sequence)  # Load the sample sequence from the database

    if not sample_sequence:
        flash("No samples found for this task. Please start again.")
        return redirect(url_for("start"))

    if page > user.num_pages:
        user.completed = True
        db.session.commit()
        return redirect(url_for("end"))

    current_sample = sample_sequence[page - 1]  # Get the current sample for this page
    
    print(f"DEBUG: Current sample: {current_sample}")
    print(f"DEBUG: Files in sample: {current_sample.get('files', [])}")
    
    # Separate GT from other files
    gt_file = None
    other_files = []
    
    for file in current_sample["files"]:
        if file["model_name"] == "gt":
            gt_file = file
        else:
            other_files.append(file)
    
    print(f"DEBUG: GT file: {gt_file}")
    print(f"DEBUG: Other files count: {len(other_files)}")
    
    # Shuffle only the non-GT files
    random.shuffle(other_files)
    
    scores = {}

    if request.method == "POST":
        # Only check scores for non-GT files
        for file in other_files:
            score_key = f'score_{file["model_name"]}_{file["file_name"]}'
            score_value = request.form.get(score_key)
            if score_value:
                # Save the score with model_name, file_name, and score
                new_score = Score(
                    user_id=user_id,
                    model_name=file["model_name"],
                    file_name=file["file_name"],
                    score=int(score_value),
                )
                db.session.add(new_score)
                db.session.commit()
                scores[score_key] = int(score_value)  # Save the score in the dictionary
            else:
                # Return early if the sample is missing a score
                flash("Please score all samples before proceeding to the next page.")
                return render_template(
                    "score.html",
                    samples=[current_sample],  # Pass the current sample
                    gt_file=gt_file,
                    other_files=other_files,
                    page=page,
                    total_pages=user.num_pages,
                    scores=scores,
                    task_type=user.task_type,
                )

        # Proceed to the next page
        session["page"] += 1

        return redirect(url_for("score"))

    return render_template(
        "score.html",
        samples=[current_sample],  # Pass the current sample
        gt_file=gt_file,
        other_files=other_files,
        page=page,
        total_pages=user.num_pages,
        scores=scores,  # Pass the scores dictionary
        task_type=user.task_type,
    )


@app.route("/end")
def end():
    if "user_id" not in session:
        return redirect(url_for("start"))

    user = db.session.get(User, session["user_id"])
    scores = {
        f"{score.model_name}_{score.file_name}": score.score for score in user.scores
    }
    session.clear()  # Clear session after survey completion
    return render_template(
        "end.html", name=user.name, task_type=user.task_type, scores=scores
    )


@app.route("/results", methods=["GET", "POST"])
@login_required
def results():
    selected_users = request.form.getlist("users") if request.method == "POST" else None
    all_users = User.query.all()

    # Prepare the query for calculating mean and standard deviation
    query = (
        db.session.query(
            User.task_type,
            Score.model_name,
            func.avg(Score.score).label("mean_score"),
            func.group_concat(Score.score, ",").label("all_scores"),
        )
        .join(Score, User.id == Score.user_id)
        .group_by(User.task_type, Score.model_name)
    )

    # Filter the query if users are selected
    if selected_users:
        query = query.filter(User.name.in_(selected_users))

    # Execute the query and process results
    results_data = query.all()
    results_summary = {}
    for task_type, model_name, mean_score, all_scores in results_data:
        scores_list = list(map(int, all_scores.split(",")))
        # stddev_score = round(stdev(scores_list), 2) if len(scores_list) > 1 else 0
        stddev_score = 1.96 * np.std(scores_list) / np.sqrt(len(scores_list))

        if task_type not in results_summary:
            results_summary[task_type] = []
        results_summary[task_type].append(
            {
                "model_name": model_name,
                "mean_score": round(mean_score, 2),
                "stddev_score": stddev_score,
            }
        )

    # Calculate individual user statistics including the number of samples rated
    user_summary = {}
    for user in all_users:
        user_task_summary = []
        num_samples = Score.query.filter_by(user_id=user.id).count()
        for task_type, model_name, mean_score, all_scores in (
            db.session.query(
                User.task_type,
                Score.model_name,
                func.avg(Score.score).label("mean_score"),
                func.group_concat(Score.score, ",").label("all_scores"),
            )
            .join(Score, User.id == Score.user_id)
            .group_by(User.task_type, Score.model_name)
            .filter(User.name == user.name)
        ):
            scores_list = list(map(int, all_scores.split(",")))
            stddev_score = round(stdev(scores_list), 2) if len(scores_list) > 1 else 0

            user_task_summary.append(
                {
                    "model_name": model_name,
                    "mean_score": round(mean_score, 2),
                    "stddev_score": stddev_score,
                }
            )

        user_summary[user.name] = {
            "task_type": user.task_type,
            "task_summary": user_task_summary,
            "num_samples": num_samples,
            "include_in_statistics": (
                user.name in selected_users if selected_users else True
            ),
        }

    return render_template(
        "results.html",
        results_summary=results_summary,
        user_summary=user_summary,
        all_users=all_users,
    )


def export_users_to_csv(users):
    si = StringIO()
    writer = csv.writer(si)

    # Write headers
    writer.writerow(["User ID", "Name", "Task Type", "Completed"])

    # Write user data
    for user in users:
        writer.writerow([user.id, user.name, user.task_type, user.completed])

    # Write score data
    writer.writerow([])
    writer.writerow(["User ID", "Model Name", "File Name", "Score"])
    for user in users:
        for score in user.scores:
            writer.writerow([user.id, score.model_name, score.file_name, score.score])

    return si.getvalue()


def export_users_to_json(users):
    users_list = []
    for user in users:
        user_data = {
            "id": user.id,
            "name": user.name,
            "task_type": user.task_type,
            "completed": user.completed,
            "scores": [
                {
                    "model_name": score.model_name,
                    "file_name": score.file_name,
                    "score": score.score,
                }
                for score in user.scores
            ],
        }
        users_list.append(user_data)

    return json.dumps(users_list, indent=4)


@app.route("/export/<string:file_type>")
def export_data(file_type):
    users = User.query.all()

    if file_type == "csv":
        csv_data = export_users_to_csv(users)
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=survey_data.csv"},
        )

    elif file_type == "json":
        json_data = export_users_to_json(users)
        return Response(
            json_data,
            mimetype="application/json",
            headers={"Content-disposition": "attachment; filename=survey_data.json"},
        )

    else:
        return "Invalid format. Please use 'csv' or 'json'."


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1234)
