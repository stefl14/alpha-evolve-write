"""Pytest configuration and fixtures."""

from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.fixture
def sample_essay_prompt():
    """Sample essay prompt for testing."""
    return "Write an essay about the impact of artificial intelligence on society."


@pytest.fixture
def sample_essay_draft():
    """Sample essay draft for testing."""
    return """
    Artificial Intelligence and Its Impact on Society

    Artificial intelligence (AI) has emerged as one of the most transformative
    technologies of our time. Its impact on society is multifaceted, affecting
    everything from healthcare to transportation, education to entertainment.

    In healthcare, AI has revolutionized diagnosis and treatment. Machine learning
    algorithms can now detect diseases in medical images with accuracy that often
    surpasses human doctors. This has led to earlier detection and better outcomes
    for patients.

    However, the rise of AI also presents challenges. Job displacement is a
    significant concern as automation becomes more sophisticated. Many traditional
    roles are at risk of being replaced by AI systems.

    In conclusion, while AI presents tremendous opportunities for societal benefit,
    we must carefully consider its implementation to ensure that its advantages are
    shared broadly and its risks are minimized.
    """.strip()


@pytest.fixture
def temp_env_file(tmp_path):
    """Create a temporary environment file for testing."""
    env_file = tmp_path / ".env"
    env_content = """
OPENAI_API_KEY=test_key
LOG_LEVEL=DEBUG
MAX_GENERATIONS=3
INITIAL_DRAFT_COUNT=2
""".strip()
    env_file.write_text(env_content)
    return env_file


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing."""
    mock_server = Mock()
    mock_server.search_papers = AsyncMock()
    mock_server.search_papers.return_value = [
        {
            "title": "The Impact of AI on Society",
            "authors": ["Smith, J.", "Doe, A."],
            "abstract": "This paper examines the societal implications of AI...",
            "url": "https://example.com/paper1",
        }
    ]
    return mock_server
