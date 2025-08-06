# 1. Use an official lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy the project files
COPY . .

# 4. Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 5. Expose the port Streamlit uses
EXPOSE 8501

# 6. Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# 7. Run the app
CMD ["streamlit", "run", "groq_rag_chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
