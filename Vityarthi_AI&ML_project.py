from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# JOB DESCRIPTION
def get_job_description():
    return """
    Required: Python, SQL, Data Analysis, Machine Learning, Excel
    Soft Skills: Communication, Problem-solving
    """

# USER INPUT
def create_resume():
    print("Enter your details:\n")
    
    name = input("Name: ")
    skills = input("Skills (comma separated): ")
    projects = input("Projects: ")
    soft_skills = input("Soft Skills: ")
    
    resume = f"""
    Name: {name}
    Skills: {skills}
    Projects: {projects}
    Soft Skills: {soft_skills}
    """
    
    return resume, name

# IDEAL RESUME GENERATOR
def generate_ideal_resume(name, job_text):
    skills = extract_skills(job_text)
    
    ideal_resume = f"""
Name: {name}

Skills:
{', '.join(skills)}

Projects:
- Developed a machine learning model using Python to analyze large datasets, improving prediction accuracy by 20%.
- Designed and implemented a data analysis dashboard using Python, SQL, and Excel to visualize key insights and support decision-making.

Soft Skills:
Communication, Problem-solving

Summary:
A results-driven candidate with strong technical expertise in Python, SQL, machine learning, and data analysis.
Experienced in building impactful projects and delivering data-driven solutions aligned with industry requirements.
"""
    return ideal_resume

# ANALYZER
def calculate_similarity(resume_text, job_text):
    documents = [resume_text.lower(), job_text.lower()]
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(documents)
    return cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

# SCORER
def calculate_score(similarity):
    return round(similarity * 100, 2)

# SKILLS
def extract_skills(text):
    skills_list = [
        "python","sql","data analysis","machine learning",
        "html","css","excel","communication",
        "teamwork","problem-solving"
    ]
    return [skill for skill in skills_list if skill in text.lower()]

def missing_skills(resume_text, job_text):
    return list(set(extract_skills(job_text)) - set(extract_skills(resume_text)))

# FEEDBACK
def generate_feedback(score, missing):
    feedback = ""

    if score > 80:
        feedback += "🔥 Excellent match! Your resume is strong.\n\n"
    elif score > 60:
        feedback += "👍 Good, but can be improved.\n\n"
    else:
        feedback += "⚠️ Resume needs improvement.\n\n"

    if missing:
        feedback += "❗ Missing Skills:\n"
        for skill in missing:
            feedback += f"- {skill}\n"
    else:
        feedback += "✅ No major skill gaps!\n"

    feedback += "\n💡 Suggestions:\n"
    feedback += "- Add missing technical skills by including tools and technologies mentioned in the job description.\n"
    feedback += "- Improve project descriptions by clearly explaining what you built, which technologies you used, and the impact or results of your work.\n"
    feedback += "- Use strong action verbs like 'developed', 'designed', 'implemented', or 'analyzed' to make your experience sound more impactful.\n"

    return feedback

# MAIN
def main():
    print("🤖 AI Resume Analyzer (Demo Version)\n")

    resume, name = create_resume()
    job = get_job_description()

    similarity = calculate_similarity(resume, job)
    score = calculate_score(similarity)

    skills = extract_skills(resume)
    missing = missing_skills(resume, job)

    # RESULTS
    print("\n📊 RESULTS")
    print("----------------------------------------")
    print(f"Resume Score: {score}%")
    print(f"Detected Skills: {', '.join(skills)}")

    # FEEDBACK
    print("\n🧠 FEEDBACK")
    print("----------------------------------------")
    print(generate_feedback(score, missing))

    # IDEAL RESUME
    print("\n🌟 IDEAL RESUME (100% MATCH EXAMPLE)")
    print("----------------------------------------")
    ideal = generate_ideal_resume(name, job)
    print(ideal)

# RUN
if __name__ == "__main__":
    main()
