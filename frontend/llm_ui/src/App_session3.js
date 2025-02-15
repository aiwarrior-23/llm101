import React, { useState, useEffect } from 'react';
import {
  Container,
  Button,
  Row,
  Col,
  Spinner,
  Form,
  Dropdown,
  DropdownButton,
  Navbar,
  Nav,
  Offcanvas,
} from 'react-bootstrap';
import Swal from 'sweetalert2';
import { generateAnswer, startNewChat, getSessions, getChatHistory } from './config';
import './App.css';

function App() {
  const [isAnswerMode, setIsAnswerMode] = useState(false);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [framework, setFramework] = useState('openai'); // default to Open AI
  const [chatHistory, setChatHistory] = useState([]); // Stores the chat history
  const [showSidebar, setShowSidebar] = useState(false); // Sidebar visibility
  const [sessionId, setSessionId] = useState(null); // Store session ID
  const [newChat, setNewChat] = useState(false); // Trigger for starting a new chat
  const [sessionsList, setSessionsList] = useState([]); // Store dynamic session values
  const [historyLoading, setHistoryLoading] = useState(false); // Spinner for chat history loading

  useEffect(() => {
    if (!newChat) return;
    const fetchSessionId = async () => {
      try {
        const response = await fetch(startNewChat, { method: 'GET' });
        const data = await response.json();
        if (response.ok) {
          setSessionId(data.session_id);
        } else {
          Swal.fire({
            icon: 'error',
            title: 'Error',
            text: data.message || 'Failed to fetch session ID.',
          });
        }
      } catch (error) {
        Swal.fire({
          icon: 'error',
          title: 'Error',
          text: 'Failed to connect to the server. Please try again later.',
        });
      }
    };

    fetchSessionId();
    setNewChat(false);
  }, [newChat]);

  useEffect(() => {
    if (showSidebar) {
      const fetchSessions = async () => {
        try {
          const response = await fetch(getSessions, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: framework }),
          });
          const data = await response.json();
          if (response.ok) {
            setSessionsList(data.sessions_list || []);
          } else {
            Swal.fire({
              icon: 'error',
              title: 'Error',
              text: data.message || 'Failed to fetch sessions.',
            });
          }
        } catch (error) {
          Swal.fire({
            icon: 'error',
            title: 'Error',
            text: 'Failed to connect to the server. Please try again later.',
          });
        }
      };

      fetchSessions();
    }
  }, [showSidebar, framework]);

  const fetchChatHistory = async (sessionName) => {
    setHistoryLoading(true);
    try {
      const response = await fetch(getChatHistory, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_name: sessionName, model_type: framework }),
      });

      const data = await response.json();
      setHistoryLoading(false);

      if (response.ok) {
        const newChatHistory = data.chat_history.map((chat) => ({
          question:
            framework === 'langchain'
              ? chat.type === 'human' && chat.content
              : chat.role === 'user' && chat.content,
          answer:
            framework === 'langchain'
              ? chat.type === 'ai' && chat.content
              : chat.role === 'assistant' && chat.content,
        }));

        setChatHistory(newChatHistory.filter((chat) => chat.question || chat.answer));
        setIsAnswerMode(true);
        setSessionId(data.session_id);
      } else {
        Swal.fire({
          icon: 'error',
          title: 'Error',
          text: data.message || 'Failed to fetch chat history.',
        });
      }
    } catch (error) {
      setHistoryLoading(false);
      Swal.fire({
        icon: 'error',
        title: 'Error',
        text: 'Failed to connect to the server. Please try again later.',
      });
    }
  };

  const handleNewChat = () => {
    setChatHistory([]);
    setIsAnswerMode(false);
    setShowSidebar(false);
    setNewChat(true);
  };

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
      const response = await fetch(generateAnswer, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          model_type: framework,
          session_id: sessionId,
        }),
      });

      const data = await response.json();
      setLoading(false);

      if (response.ok) {
        setChatHistory([
          ...chatHistory,
          { question: text, answer: data.answer || 'No response available.' },
        ]);
        setText('');
      } else {
        Swal.fire({
          icon: 'error',
          title: 'Error',
          text: data.message || 'Failed to get an answer.',
        });
      }
    } catch (error) {
      setLoading(false);
      Swal.fire({
        icon: 'error',
        title: 'Error',
        text: 'Failed to connect to the server. Please try again later.',
      });
    }
  };

  return (
    <div>
      <Navbar expand="lg" className="custom-navbar">
        <Container>
          <Navbar.Brand className="d-flex align-items-center brand-name">
            <i className="fas fa-robot me-2"></i> Auto LLM
            <Button
              variant="outline-light"
              onClick={() => setShowSidebar(!showSidebar)}
              className="ms-3 sidebar-toggle-btn"
            >
              ☰
            </Button>
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto me-3">
              <DropdownButton
                variant="primary"
                title={`Model: ${framework.charAt(0).toUpperCase() + framework.slice(1)}`}
                onSelect={setFramework}
                className="custom-dropdown"
              >
                <Dropdown.Item eventKey="openai">
                  <i className="fas fa-brain me-2"></i> Open AI
                </Dropdown.Item>
                <Dropdown.Item eventKey="langchain">
                  <i className="fas fa-link me-2"></i> Langchain
                </Dropdown.Item>
                <Dropdown.Item eventKey="llamaindex">
                  <i className="fas fa-chart-line me-2"></i> Llama Index
                </Dropdown.Item>
              </DropdownButton>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Offcanvas
        show={showSidebar}
        onHide={() => setShowSidebar(false)}
        placement="start"
        className="custom-sidebar"
      >
        <Offcanvas.Header closeButton>
          <Offcanvas.Title className="fs-4 text-primary">Menu</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body className="p-0">
          <Nav className="flex-column">
            <Nav.Link
              className="sidebar-link p-3 text-dark fw-bold"
              onClick={handleNewChat}
            >
              <i className="fas fa-plus-circle me-2 text-success"></i> Start New Chat
            </Nav.Link>
            <div className="p-3 bg-light border-top border-bottom fw-bold">Chat History</div>
            <Nav className="ms-3">
              {sessionsList.length > 0 ? (
                sessionsList.map((session, index) => (
                  <Nav.Link
                    key={index}
                    className="sidebar-link p-3 text-dark"
                    onClick={() => fetchChatHistory(session)}
                  >
                    <i className="fas fa-comments me-2 text-info"></i>
                    {session}
                  </Nav.Link>
                ))
              ) : (
                <Nav.Link className="sidebar-link p-3 text-muted">
                  <i className="fas fa-exclamation-circle me-2"></i> No history available
                </Nav.Link>
              )}
            </Nav>
          </Nav>
        </Offcanvas.Body>
      </Offcanvas>

      <Container style={{ marginTop: '60px' }} className="mt-3">
        {historyLoading && (
          <Row className="justify-content-center">
            <Col xs="auto">
              <Spinner animation="border" />
            </Col>
          </Row>
        )}

        {chatHistory.length > 0 && (
          <Row className="justify-content-center mb-0">
            <Col
              md={10}
              style={{
                maxHeight: '500px',
                overflowY: 'auto',
                border: '1px solid #ccc',
                padding: '15px',
                borderRadius: '5px',
              }}
            >
              {chatHistory.map((chat, index) => (
                <React.Fragment key={index}>
                  {chat.question && (
                    <Row className="mb-1">
                      <Col xs={6} className="ms-auto text-end">
                        <div
                          className="p-3 bg-primary text-white rounded"
                          style={{
                            display: 'inline-block',
                            maxWidth: '80%',
                          }}
                        >
                          <strong>You:</strong> {chat.question}
                        </div>
                      </Col>
                    </Row>
                  )}
                  {chat.answer && (
                    <Row className="mb-3">
                      <Col xs={6}>
                        <div
                          className="p-3 bg-success text-white rounded"
                          style={{
                            display: 'inline-block',
                            maxWidth: '80%',
                          }}
                        >
                          <strong>AI:</strong> {chat.answer}
                        </div>
                      </Col>
                    </Row>
                  )}
                </React.Fragment>
              ))}
            </Col>
          </Row>
        )}

        <Row className="justify-content-center mt-3">
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
            <Button
              variant="primary"
              onClick={handleButtonClick}
              disabled={loading}
              style={{ borderRadius: '0' }}
            >
              Submit
            </Button>
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;