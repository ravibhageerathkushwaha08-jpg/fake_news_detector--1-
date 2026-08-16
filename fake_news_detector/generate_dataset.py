"""
generate_dataset.py
--------------------
Creates a sample dataset (data/news.csv) for the Fake News Detection project.

NOTE: This is a small, synthetically-built dataset meant ONLY for learning
purposes so beginners can run the full project end-to-end without needing
to download anything from the internet. For a real project, replace
data/news.csv with a proper dataset such as the Kaggle "Fake and Real News"
dataset (https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
"""

import csv
import random

random.seed(42)

# ---- Building blocks for REAL (factual / formal style) news ----
real_subjects = [
    "The Reserve Bank of India", "The Prime Minister", "Scientists at IIT Bombay",
    "The Ministry of Health", "Local authorities", "The World Health Organization",
    "Researchers at Harvard University", "The Election Commission",
    "The state government", "NASA", "The United Nations", "The Finance Ministry",
    "City officials", "The Supreme Court", "Economists", "The company's board",
    "University researchers", "The central government", "Health officials",
    "The company Tesla",
]

real_actions = [
    "announced a new policy to improve infrastructure spending",
    "released its quarterly report showing steady growth",
    "confirmed that the new vaccine trial met safety standards",
    "held a press conference to discuss the upcoming budget",
    "published a study on climate change trends in the region",
    "approved funding for a new public transportation project",
    "reported a decline in unemployment figures for the last quarter",
    "signed a trade agreement with neighboring countries",
    "launched a new satellite to monitor weather patterns",
    "issued guidelines for road safety during the festival season",
    "confirmed the results of the recent state elections",
    "announced scholarships for students from low income families",
    "opened a new research center focused on renewable energy",
    "released data showing improvement in literacy rates",
    "began construction on a new hospital in the district",
    "clarified that the reported rumors about the merger are false",
    "conducted an audit of the state's healthcare spending",
    "will hold a public consultation on the new education policy",
    "confirmed a drop in inflation for the third straight month",
    "inaugurated a new bridge connecting the two districts",
]

real_sources = [
    "according to an official statement",
    "as reported by Reuters",
    "based on data from the Ministry of Statistics",
    "as confirmed in a press release",
    "according to the official government website",
    "as stated in the annual report",
    "based on figures published by the central bank",
    "according to officials who spoke on condition of anonymity",
]

# ---- Building blocks for FAKE (sensational / clickbait style) news ----
fake_subjects = [
    "Doctors", "Scientists", "A secret government report", "This one weird trick",
    "Local mom", "Anonymous sources", "A viral WhatsApp message",
    "Shocking new leaked video", "Big Pharma", "The illuminati", "Aliens",
    "A miracle fruit", "Government insiders", "A secret society",
    "This shocking discovery", "Celebrities", "A mysterious billionaire",
    "Hidden cameras", "A banned ancient remedy", "Top secret documents",
]

fake_actions = [
    "reveal the truth they don't want you to know about vaccines",
    "confirm that drinking this juice cures cancer overnight",
    "prove that the moon landing was completely fake",
    "expose a secret plan to microchip every citizen by next year",
    "show that eating this fruit melts belly fat in 3 days",
    "warn that your phone is secretly recording everything you say",
    "claim that the earth is actually flat and NASA is hiding it",
    "reveal that a celebrity secretly died years ago and was replaced",
    "prove that 5G towers are spreading a dangerous new disease",
    "show shocking footage that the mainstream media refuses to air",
    "confirm that a miracle pill can make you lose 10kg in a week",
    "reveal a hidden message found in an old currency note",
    "expose how a common vegetable can cure every disease known to man",
    "warn that a popular soft drink is secretly poisoning millions",
    "prove aliens are hiding inside a mountain and the government knows",
    "claim a famous actor secretly controls the world's banks",
    "reveal that schools are hiding a shocking secret from parents",
    "show that a common household item is actually deadly radioactive",
    "confirm that time travel was achieved in a secret lab last year",
    "expose a plot to replace world leaders with robotic doubles",
]

fake_flourishes = [
    "Share this before it gets deleted!!!",
    "You won't believe what happens next.",
    "This is being hidden from the public!",
    "Doctors hate this simple trick.",
    "Wake up, people, before it's too late!",
    "This changes everything we thought we knew.",
    "The mainstream media won't tell you this.",
    "Forward to everyone you know right now!",
    "This will blow your mind.",
    "They don't want you to see this video.",
]


def make_real_sentence():
    subj = random.choice(real_subjects)
    action = random.choice(real_actions)
    src = random.choice(real_sources)
    return f"{subj} {action}, {src}."


def make_fake_sentence():
    subj = random.choice(fake_subjects)
    action = random.choice(fake_actions)
    flourish = random.choice(fake_flourishes)
    return f"{subj} {action}. {flourish}"


def build_dataset(n_per_class=300):
    rows = []
    seen = set()
    while len([r for r in rows if r[1] == "REAL"]) < n_per_class:
        s = make_real_sentence()
        if s not in seen:
            seen.add(s)
            rows.append((s, "REAL"))
    while len([r for r in rows if r[1] == "FAKE"]) < n_per_class:
        s = make_fake_sentence()
        if s not in seen:
            seen.add(s)
            rows.append((s, "FAKE"))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build_dataset(n_per_class=300)
    with open("data/news.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to data/news.csv")
