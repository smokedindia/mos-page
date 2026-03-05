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
PROJECT_NAME = "2026_TMLR_videogen"


app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{PROJECT_NAME}.db"
db = SQLAlchemy(app)

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
    file_name = db.Column(db.String(100), nullable=False)
    pair_type = db.Column(db.String(50), nullable=False)  # single_vs_multi or naive_vs_multi
    model1_name = db.Column(db.String(50), nullable=False)  # First model in comparison
    model2_name = db.Column(db.String(50), nullable=False)  # Second model in comparison
    # Store actual winner name or "tie"
    text_consistency = db.Column(db.String(20), nullable=False)  # model name or "tie"
    audio_video_alignment = db.Column(db.String(20), nullable=False)  # model name or "tie"
    overall_quality = db.Column(db.String(20), nullable=False)  # model name or "tie"


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
            # Sample directory structure
            models = os.listdir(os.path.join("static", "samples", PROJECT_NAME, "javisdit"))
            SAMPLE_DIRECTORIES = {model: f"static/samples/{PROJECT_NAME}/javisdit/{model}" for model in models}

            # Load prompts from CSV
            prompts = {}
            prompt_csv_path = os.path.join("static", "samples", PROJECT_NAME, "prompt.csv")
            if os.path.exists(prompt_csv_path):
                with open(prompt_csv_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    for row in csv_reader:
                        prompts[row['video_path']] = row['text']

            samples = []

            # Load sample filenames from one directory
            model_to_use_for_listing = "single"  # Choose one model directory to list the filenames
            model_path = SAMPLE_DIRECTORIES[model_to_use_for_listing]
            file_names = [
                file_name for file_name in os.listdir(model_path) if file_name.endswith(".mp4")
            ]

            # Create pairwise comparisons: single vs multi, naive vs multi
            for file_name in file_names:
                # Get prompt text
                prompt_text = prompts.get(file_name, "")
                
                # Check if all three models exist for this sample
                single_path = os.path.join(SAMPLE_DIRECTORIES.get("single", ""), file_name)
                naive_path = os.path.join(SAMPLE_DIRECTORIES.get("naive", ""), file_name)
                multi_path = os.path.join(SAMPLE_DIRECTORIES.get("multi", ""), file_name)
                
                # Pair 1: single vs multi
                if os.path.exists(single_path) and os.path.exists(multi_path):
                    samples.append({
                        "file_name": file_name,
                        "prompt": prompt_text,
                        "pair_type": "single_vs_multi",
                        "left": {
                            "model_name": "single",
                            "file_name": file_name,
                            "file_path": single_path.replace("static/", ""),
                        },
                        "right": {
                            "model_name": "multi",
                            "file_name": file_name,
                            "file_path": multi_path.replace("static/", ""),
                        }
                    })
                
                # Pair 2: naive vs multi
                if os.path.exists(naive_path) and os.path.exists(multi_path):
                    samples.append({
                        "file_name": file_name,
                        "prompt": prompt_text,
                        "pair_type": "naive_vs_multi",
                        "left": {
                            "model_name": "naive",
                            "file_name": file_name,
                            "file_path": naive_path.replace("static/", ""),
                        },
                        "right": {
                            "model_name": "multi",
                            "file_name": file_name,
                            "file_path": multi_path.replace("static/", ""),
                        }
                    })

            # New user, create a new record
            num_pages = 20
            # Filter samples by the selected task_type (comparison pair)
            filtered_samples = [s for s in samples if s["pair_type"] == task_type]
            selected_samples = random.sample(
                filtered_samples, min(num_pages, len(filtered_samples))
            )  # Randomly select samples
            sample_sequence = json.dumps(
                selected_samples
            )  # Store the sequence as a JSON string

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
    user = User.query.get(user_id)
    page = session.get("page", 1)
    sample_sequence = json.loads(
        user.sample_sequence
    )  # Load the sample sequence from the database

    if page > user.num_pages:
        user.completed = True
        db.session.commit()
        return redirect(url_for("end"))

    current_sample = sample_sequence[page - 1]  # Get the current sample for this page
    
    scores = {}

    if request.method == "POST":
        # Get which models were actually displayed from hidden form fields
        left_model = request.form.get("left_model")
        right_model = request.form.get("right_model")
        
        # Get the three criteria scores
        text_consistency = request.form.get("text_consistency")
        audio_video_alignment = request.form.get("audio_video_alignment")
        overall_quality = request.form.get("overall_quality")
        
        if text_consistency and audio_video_alignment and overall_quality and left_model and right_model:
            # Convert position-based votes to model name votes
            def convert_vote(vote, left_model, right_model):
                if vote == "tie":
                    return "tie"
                elif vote == "left_win":
                    return left_model
                elif vote == "right_win":
                    return right_model
            
            text_consistency_winner = convert_vote(text_consistency, left_model, right_model)
            audio_video_alignment_winner = convert_vote(audio_video_alignment, left_model, right_model)
            overall_quality_winner = convert_vote(overall_quality, left_model, right_model)
            
            # Save the score
            new_score = Score(
                user_id=user_id,
                file_name=current_sample["file_name"],
                pair_type=current_sample["pair_type"],
                model1_name=left_model,
                model2_name=right_model,
                text_consistency=text_consistency_winner,
                audio_video_alignment=audio_video_alignment_winner,
                overall_quality=overall_quality_winner,
            )
            db.session.add(new_score)
            db.session.commit()
            
            # Proceed to the next page
            session["page"] += 1
            return redirect(url_for("score"))
        else:
            # Return early if any criteria is missing
            flash("Please complete all three scoring criteria before proceeding.")
            # Re-apply random swap for display
            if random.random() < 0.5:
                current_sample["left"], current_sample["right"] = current_sample["right"], current_sample["left"]
            return render_template(
                "score.html",
                sample=current_sample,
                page=page,
                total_pages=user.num_pages,
                scores=scores,
                task_type=user.task_type,
            )

    # For GET requests, randomly swap left and right videos (50% chance)
    if random.random() < 0.5:
        current_sample["left"], current_sample["right"] = current_sample["right"], current_sample["left"]

    return render_template(
        "score.html",
        sample=current_sample,
        page=page,
        total_pages=user.num_pages,
        scores=scores,
        task_type=user.task_type,
    )


@app.route("/end")
def end():
    if "user_id" not in session:
        return redirect(url_for("start"))

    user = User.query.get(session["user_id"])
    session.clear()  # Clear session after survey completion
    return render_template(
        "end.html", name=user.name, task_type=user.task_type
    )


@app.route("/results", methods=["GET", "POST"])
@login_required
def results():
    selected_users = request.form.getlist("users") if request.method == "POST" else None
    all_users = User.query.all()

    # Get all scores
    query = db.session.query(Score).join(User, User.id == Score.user_id)
    
    # Filter the query if users are selected
    if selected_users:
        query = query.filter(User.name.in_(selected_users))
    
    all_scores = query.all()
    
    # Calculate win rates for each pair type and criterion
    results_summary = {}
    
    for pair_type in ["single_vs_multi", "naive_vs_multi"]:
        pair_scores = [s for s in all_scores if s.pair_type == pair_type]
        
        if not pair_scores:
            continue
        
        # Get all unique models for this pair type
        all_models = set()
        for score in pair_scores:
            all_models.add(score.model1_name)
            all_models.add(score.model2_name)
        all_models = sorted(list(all_models))
        
        # Calculate stats by model name for each criterion
        def calculate_model_stats(criterion_name):
            model_stats = {}
            for model in all_models:
                model_stats[model] = {"wins": 0, "ties": 0, "losses": 0}
            
            for score in pair_scores:
                winner = getattr(score, criterion_name)
                model1 = score.model1_name
                model2 = score.model2_name
                
                if winner == "tie":
                    model_stats[model1]["ties"] += 1
                    model_stats[model2]["ties"] += 1
                else:
                    # winner is the model name
                    model_stats[winner]["wins"] += 1
                    loser = model1 if winner == model2 else model2
                    model_stats[loser]["losses"] += 1
            
            # Calculate percentages
            for model in model_stats:
                total = model_stats[model]["wins"] + model_stats[model]["ties"] + model_stats[model]["losses"]
                if total > 0:
                    model_stats[model]["wins_pct"] = round(100 * model_stats[model]["wins"] / total, 1)
                    model_stats[model]["ties_pct"] = round(100 * model_stats[model]["ties"] / total, 1)
                    model_stats[model]["losses_pct"] = round(100 * model_stats[model]["losses"] / total, 1)
                else:
                    model_stats[model]["wins_pct"] = 0
                    model_stats[model]["ties_pct"] = 0
                    model_stats[model]["losses_pct"] = 0
            
            return model_stats
        
        results_summary[pair_type] = {
            "text_consistency": calculate_model_stats("text_consistency"),
            "audio_video_alignment": calculate_model_stats("audio_video_alignment"),
            "overall_quality": calculate_model_stats("overall_quality"),
            "total_comparisons": len(pair_scores)
        }
    
    # Calculate individual user statistics
    user_summary = {}
    for user in all_users:
        user_scores = [s for s in all_scores if s.user_id == user.id]
        num_samples = len(user_scores)
        
        user_results = {}
        for pair_type in ["single_vs_multi", "naive_vs_multi"]:
            pair_scores = [s for s in user_scores if s.pair_type == pair_type]
            if pair_scores:
                # Get all unique models for this pair type
                all_models = set()
                for score in pair_scores:
                    all_models.add(score.model1_name)
                    all_models.add(score.model2_name)
                all_models = sorted(list(all_models))
                
                # Calculate stats by model name for each criterion
                def calculate_model_stats(criterion_name):
                    model_stats = {}
                    for model in all_models:
                        model_stats[model] = {"wins": 0, "ties": 0, "losses": 0}
                    
                    for score in pair_scores:
                        winner = getattr(score, criterion_name)
                        model1 = score.model1_name
                        model2 = score.model2_name
                        
                        if winner == "tie":
                            model_stats[model1]["ties"] += 1
                            model_stats[model2]["ties"] += 1
                        else:
                            # winner is the model name
                            model_stats[winner]["wins"] += 1
                            loser = model1 if winner == model2 else model2
                            model_stats[loser]["losses"] += 1
                    
                    # Calculate percentages
                    for model in model_stats:
                        total = model_stats[model]["wins"] + model_stats[model]["ties"] + model_stats[model]["losses"]
                        if total > 0:
                            model_stats[model]["wins_pct"] = round(100 * model_stats[model]["wins"] / total, 1)
                            model_stats[model]["ties_pct"] = round(100 * model_stats[model]["ties"] / total, 1)
                            model_stats[model]["losses_pct"] = round(100 * model_stats[model]["losses"] / total, 1)
                        else:
                            model_stats[model]["wins_pct"] = 0
                            model_stats[model]["ties_pct"] = 0
                            model_stats[model]["losses_pct"] = 0
                    
                    return model_stats
                
                user_results[pair_type] = {
                    "text_consistency": calculate_model_stats("text_consistency"),
                    "audio_video_alignment": calculate_model_stats("audio_video_alignment"),
                    "overall_quality": calculate_model_stats("overall_quality"),
                    "total_comparisons": len(pair_scores)
                }
        
        user_summary[user.name] = {
            "task_type": user.task_type,
            "results": user_results,
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
    writer.writerow(["User ID", "File Name", "Pair Type", "Model 1", "Model 2", "Text Consistency Winner", "Audio-Video Alignment Winner", "Overall Quality Winner"])
    for user in users:
        for score in user.scores:
            writer.writerow([
                user.id,
                score.file_name,
                score.pair_type,
                score.model1_name,
                score.model2_name,
                score.text_consistency,
                score.audio_video_alignment,
                score.overall_quality
            ])

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
                    "file_name": score.file_name,
                    "pair_type": score.pair_type,
                    "model1_name": score.model1_name,
                    "model2_name": score.model2_name,
                    "text_consistency_winner": score.text_consistency,
                    "audio_video_alignment_winner": score.audio_video_alignment,
                    "overall_quality_winner": score.overall_quality,
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
    app.run(host='0.0.0.0', port=5000)
