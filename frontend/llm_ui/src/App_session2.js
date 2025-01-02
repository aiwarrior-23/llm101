import React, { useState } from 'react';
import { Container, Button, Row, Col, Spinner, Form, Dropdown, DropdownButton } from 'react-bootstrap';
import Swal from 'sweetalert2';
import './App.css'; // Import custom CSS file

function App() {
  const [isAnswerMode, setIsAnswerMode] = useState(false);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [framework, setFramework] = useState('openai'); // default to Open AI
  const [chatHistory, setChatHistory] = useState([]); // Stores the chat history

  const endpoint = 'http://localhost:8000/generate_answer';

  const handleButtonClick = async () => {
    if (!text.trim()) {
      Swal.fire({
        icon: 'warning',
        title: 'Input Required',
        text: 'Please enter a question to get an answer.',
      });
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, model_type: framework }),
      });

      const data = await response.json();
      setLoading(false);

      if (response.ok) {
        setIsAnswerMode(true);
        setChatHistory([...chatHistory, { question: text, answer: data.answer || 'This is the answer paragraph.' }]);
        setText(''); // Clear input
      } else {
        Swal.fire({
          icon: 'error',
          title: 'Oops...',
          text: data.message || 'Something went wrong. Please try again.',
        });
      }
    } catch (error) {
      setLoading(false);
      Swal.fire({
        icon: 'error',
        title: 'Error',
        text: 'Failed to fetch the answer. Please check your internet connection.',
      });
    }
  };

  const handleFrameworkChange = (newFramework) => {
    if (newFramework !== framework) {
      Swal.fire({
        icon: 'question',
        title: 'Change Framework',
        text: 'Are you sure you want to switch models? This will reset the conversation history.',
        showCancelButton: true,
        confirmButtonText: 'Yes, change it!',
        cancelButtonText: 'No, keep current',
      }).then((result) => {
        if (result.isConfirmed) {
          setFramework(newFramework);
          setChatHistory([]); // Reset chat history
          setIsAnswerMode(false); // Exit answer mode
        }
      });
    }
  };

  return (
    <Container fluid className="mt-5">
      {loading && (
        <Row className="justify-content-center">
          <Col xs="auto">
            <Spinner animation="border" />
          </Col>
        </Row>
      )}

      <Row className="justify-content-center mb-4">
        <Col xs="auto" className="text-center">
          <h1 style={{ fontFamily: 'Poppins, sans-serif', fontWeight: '700', fontSize: '2.5rem', color: '#5A67D8', marginTop: '20px' }}>
            Auto LLM Chatbot
          </h1>
        </Col>
      </Row>

      {/* Conditionally render the chat interface only if there is chat history */}
      {chatHistory.length > 0 && (
        <Row className="justify-content-center mb-0">
          <Col md={10} style={{ maxHeight: '500px', overflowY: 'auto', border: '1px solid #ccc', padding: '15px', borderRadius: '5px' }}>
            {chatHistory.map((chat, index) => (
              <React.Fragment key={index}>
                {/* User Message */}
                <Row className="mb-1">
                  <Col xs={6} className="ms-auto text-end">
                    <div className="p-3 bg-primary text-white rounded" style={{ display: 'inline-block', maxWidth: '80%' }}>
                      <strong>You:</strong> {chat.question}
                    </div>
                  </Col>
                </Row>
                {/* AI Message */}
                <Row className="mb-3">
                  <Col xs={6}>
                    <div className="p-3 bg-success text-white rounded" style={{ display: 'inline-block', maxWidth: '80%' }}>
                      <strong>AI:</strong> {chat.answer}
                    </div>
                  </Col>
                </Row>
              </React.Fragment>
            ))}
          </Col>
        </Row>
      )}

      <Row className="justify-content-center mt-0">
        <Col md={8} className="pe-0">
          <Form.Control
            type="text"
            placeholder="Ask a question..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={loading}
          />
        </Col>
        <Col md="auto" className="px-0">
          <Button variant="primary" onClick={handleButtonClick} disabled={loading} style={{ borderRadius: '0' }}>
            Submit
          </Button>
        </Col>
        <Col md="auto" className="ps-0">
          <DropdownButton
            variant="secondary"
            title={`Model: ${framework.charAt(0).toUpperCase() + framework.slice(1)}`}
            onSelect={handleFrameworkChange}
            className="custom-dropdown"
          >
            <Dropdown.Item eventKey="openai">Open AI</Dropdown.Item>
            <Dropdown.Item eventKey="langchain">Langchain</Dropdown.Item>
            <Dropdown.Item eventKey="llamaindex">Llama Index</Dropdown.Item>
          </DropdownButton>
        </Col>
      </Row>
    </Container>
  );
}

export default App;