import os
import random
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Models from the image
models = ["diffv2s", "gt", "inteli", "ltbs", "ours", "svts"]


# Function to create a dummy database
def create_dummy_db(db_name):
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_name}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db = SQLAlchemy(app)

    # Define models
    class User(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(50), unique=True, nullable=False)
        task_type = db.Column(db.String(50), nullable=False)
        completed = db.Column(db.Boolean, default=False)
        scores = db.relationship("Score", backref="user", lazy=True)

    class Score(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
        model_name = db.Column(db.String(100), nullable=False)
        file_name = db.Column(db.String(100), nullable=False)
        score = db.Column(db.Integer, nullable=False)

    with app.app_context():
        db.create_all()

        # Create some dummy users
        users = [
            User(
                name=f"User_{j}",
                task_type=random.choice(["MOS", "CMOS"]),
                completed=True,
            )
            for j in range(1, 6)  # 5 users per database
        ]

        db.session.add_all(users)
        db.session.commit()

        # Create dummy scores for each user
        for user in users:
            for model in models:
                # Assume each model has 2 sample files for simplicity
                for sample_id in range(1, 3):
                    score = Score(
                        user_id=user.id,
                        model_name=model,
                        file_name=f"sample{sample_id}.wav",
                        score=random.randint(1, 5),
                    )
                    db.session.add(score)

        db.session.commit()

    print(f"Database {db_name} created with dummy data.")


# Create 10 databases
db_name = "survey.db"
create_dummy_db(db_name)
