import streamlit as st

def about_me_page():
    # Title
    st.title("👨‍💻 About Me - Omkar Rajesh Shinde")

    # Intro
    st.markdown("""
    Hello! I'm **Omkar Rajesh Shinde**, a versatile **AI/ML Engineer** specializing in **Production MLOps, Generative AI, and Full-Stack Development**.
    I combine robust core CS fundamentals with hands-on experience in building scalable, intelligent applications.
    
    My focus is on transforming complex data challenges into functional, elegant solutions ready for production environments.
    
    ---
    """)

    # Technical Skills (Using your categorized list)
    st.subheader("⚡ Technical Skills")
    st.markdown("""
    - **Languages:** Python, TypeScript, JavaScript, SQL, C++, Java 
    - **AI & GenAI:** PyTorch, TensorFlow, LLaMA-3, Hugging Face, Stable Diffusion, GANs, OpenCV 
    - **MLOps & Cloud:** Docker, Kubernetes (Minikube), AWS (EC2/S3), GitHub Actions, MLflow, Prometheus, Grafana, FastAPI
    - **Web Development:** Next.js, React.js, Flask, MongoDB, Tailwind CSS, Node.js
    - **Core Concepts:** Data Structures & Algorithms (300+ LeetCode), System Design, Computer Vision, NLP, Microservices Architecture
    
    ---
    """)

    # Projects Section
    st.subheader("🚀 Featured Projects")
    st.markdown("""
    - **[Real-Time Fraud MLOps Pipeline](https://frauddetectionmlops.streamlit.app/)** – Production-ready system using Kubernetes/Docker, achieving 99.9% availability and 92% Recall on imbalanced fraud data.
    - **[Multi-Modal Generative AI Solution](https://github.com/Omkar90492/MindCanvas--Multi-Modal-Generative-AI-System)** – Pioneers AI generation (Stable Diffusion) personalized via ResNet-50 and NLP, with sub-200ms API response time.
    - **[Intelligent Resume Ranking Microservice](https://github.com/Omkar90492/Smart-Hire-Platform)** – Full-Stack platform (Next.js/Flask/MongoDB) with an 88% accurate TF-IDF ranking algorithm.
    
    ---
    """)

    # Achievements & Links
    st.subheader("🌐 Connect and Achievements")
    st.markdown("""
    - **DSA Proficiency:** Solved **300+ DSA problems** on LeetCode. 
    - **Top 8% Global Recognition:** Recognized among the Top 8% of 905 Interns (1M1B) for reducing CO2 emissions by 0.58 MT.
    - **LinkedIn:** [linkedin.com/in/omkar-shinde-009787251](https://www.linkedin.com/in/omkar-shinde-009787251/)
    - **GitHub:** [github.com/Omkar90492](https://github.com/Omkar90492)
    - **Portfolio:** [omkar-shinde.vercel.app](https://omkar-shinde.vercel.app/)
    """)
    
    # Note: If your actual project repo links are different, replace the placeholder links above!