Welcome to the PDF Q&A Chatbot! This application allows you to interact with the content of your PDF documents, enabling you to ask questions and receive answers based on the document's content.

🚀 Demo

Experience the live application here:

🔗 https://pdfqnachatbot-q2mhdk7oekzsxyxm3jxvbv.streamlit.app/

🧠 Features

PDF Upload: Upload your PDF documents directly into the app.

Interactive Q&A: Ask questions related to the content of the uploaded PDF.

Conversational Interface: Engage in a chat-like interface for a seamless experience.

Contextual Responses: Receive answers grounded in the document's content.

🛠️ How It Works

Upload PDF: Begin by uploading your PDF document using the provided interface.

Ask Questions: Once the document is processed, type your questions into the chat input.

Receive Answers: The app processes your query and provides an answer based on the document's content.

📦 Requirements

To run this application locally, ensure you have the following Python packages installed:

pip install streamlit PyPDF2 langchain


streamlit: Framework for building the web application.

PyPDF2: Library to extract text from PDF files.

langchain: Toolkit for building language model applications.

🧪 Local Usage

Clone this repository to your local machine.

Navigate to the project directory.

Run the Streamlit app:

streamlit run app.py


Open the provided local URL in your browser to interact with the application.

🔐 Security & Privacy

No Data Storage: The application does not store any of your uploaded PDFs or chat interactions. Once the session ends, all data is discarded.

Local Processing: All processing occurs locally on your machine, ensuring your data remains private.

🛠️ Troubleshooting

PDF Upload Issues: Ensure your PDF is not encrypted or password-protected.

Large PDFs: For large documents, processing might take longer. Please be patient.

Dependencies: Ensure all required Python packages are installed and up-to-date.

📄 License

This project is open-source and available under the MIT License.

