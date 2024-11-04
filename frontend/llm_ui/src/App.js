import React, { useState } from 'react';
import { Container, Button, Row, Col, Spinner, Form } from 'react-bootstrap';
import { CKEditor } from '@ckeditor/ckeditor5-react';
import { ClassicEditor, Bold, Essentials, Italic, Mention, Paragraph, Undo } from 'ckeditor5';
import Swal from 'sweetalert2';
import ReactMarkdown from 'react-markdown';

import 'ckeditor5/ckeditor5.css';

function App() {
  const [isAnswerMode, setIsAnswerMode] = useState(false);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [framework, setFramework] = useState('openai'); // default to Open AI

  // Map framework to endpoint
  const endpoints = {
    openai: '/generate_answer',
    langchain: '/generate_answer_using_langchain',
    llamainindex: '/generate_answer_using_llama_index',
  };

  const handleButtonClick = async () => {
    if (!isAnswerMode) {
      // When "Get Answer" is clicked
      setLoading(true); // Show spinner
      try {
        const endpoint = `http://localhost:8000${endpoints[framework]}`;
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: text }),
        });

        const data = await response.json();
        setLoading(false); // Hide spinner

        if (response.ok) {
          setIsAnswerMode(true); // Show answer mode
          setText(data.answer || 'This is the answer paragraph.');
        } else {
          Swal.fire({
            icon: 'error',
            title: 'Oops...',
            text: data.message || 'Something went wrong. Please try again.',
          });
        }
      } catch (error) {
        setLoading(false); // Hide spinner
        Swal.fire({
          icon: 'error',
          title: 'Error',
          text: 'Failed to fetch the answer. Please check your internet connection.',
        });
      }
    } else {
      // When "Go Back" is clicked
      setIsAnswerMode(false);
      setText('');
    }
  };

  return (
    <Container className="mt-5">
      {/* Full-Page Loading Spinner */}
      {loading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: 'rgba(255, 255, 255, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <Spinner animation="border" />
        </div>
      )}

      {/* Title Row */}
      <Row className="justify-content-center mb-4">
        <Col xs="auto" className="text-center">
          <h1 style={{ fontFamily: 'Poppins, sans-serif', fontWeight: '700', fontSize: '2.5rem', color: '#5A67D8', marginTop: '20px' }}>
            Auto LLM
          </h1>
        </Col>
      </Row>

      {/* CKEditor or Answer Display Row */}
      <Row className="justify-content-center">
        <Col md={12}>
          {isAnswerMode ? (
            <div>
              <ReactMarkdown>{text}</ReactMarkdown>
              <p style={{ fontFamily: 'Arial, sans-serif', fontStyle: 'italic', fontSize: '10px', color:'brown' }}>
                Generated answer using <strong>{framework}</strong>
              </p>
            </div>
          ) : (
            <CKEditor
              editor={ClassicEditor}
              config={{
                toolbar: {
                  items: ['undo', 'redo', '|', 'bold', 'italic'],
                },
                plugins: [Bold, Essentials, Italic, Mention, Paragraph, Undo],
              }}
              data="<p>Ask a question...</p>"
              onChange={(event, editor) => {
                const data = editor.getData();
                setText(data);
              }}
            />
          )}
        </Col>
      </Row>

      {/* Framework Selection Row - Only Show When Not in Answer Mode */}
      {!isAnswerMode && (
        <Row className="mb-4 mt-4">
          <Col xs="auto">
            <h6>Select Framework</h6>
            <Form.Check
              type="radio"
              label="Open AI"
              name="framework"
              value="openai"
              checked={framework === 'openai'}
              onChange={() => setFramework('openai')}
            />
            <Form.Check
              type="radio"
              label="Langchain"
              name="framework"
              value="langchain"
              checked={framework === 'langchain'}
              onChange={() => setFramework('langchain')}
            />
            <Form.Check
              type="radio"
              label="Llama Index"
              name="framework"
              value="llamainindex"
              checked={framework === 'llamainindex'}
              onChange={() => setFramework('llamainindex')}
            />
          </Col>
        </Row>
      )}

      {/* Button Row */}
      <Row className="justify-content-center mt-4">
        <Button variant="primary" onClick={handleButtonClick} style={{ width: "50%" }} disabled={loading}>
          {isAnswerMode ? 'Go Back' : 'Get Answer'}
        </Button>
      </Row>
    </Container>
  );
}

export default App;
