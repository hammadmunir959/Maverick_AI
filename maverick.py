"""
Maverick - A sophisticated AI chatbot with multi-source web search capabilities
Built with LangChain for professional-grade conversational AI

Author: AI Assistant
Version: 2.1.0
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.memory import ConversationBufferWindowMemory  
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maverick.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
class Config:
    """Configuration constants for Maverick"""
    MAX_SEARCH_RESULTS = 5
    MAX_CONVERSATION_HISTORY = 50
    DEFAULT_MODEL = os.getenv('GROQ_MODEL', "mistral-saba-24b")
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    SCRAPINGDOG_API_KEY = os.getenv('SCRAPINGDOG_API_KEY')
    
    # API URLs
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    DDG_BASE_URL = "https://api.duckduckgo.com/"
    SERPER_BASE_URL = "https://google.serper.dev/search"
    SCRAPINGDOG_BASE_URL = "https://api.scrapingdog.com/google"

class SearchEngine(Enum):
    """Supported search engines"""
    DUCKDUCKGO = "duckduckgo"
    SERPER = "serper"
    SCRAPINGDOG = "scrapingdog"

class SearchDecision(BaseModel):
    """Structured output for search decision"""
    should_search: bool = Field(description="Whether web search is needed")
    search_query: Optional[str] = Field(default=None, description="Optimized search query if search is needed")
    reasoning: str = Field(description="Brief explanation of the decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision (0-1)")

@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    snippet: str
    url: str = ""
    source: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'snippet': self.snippet,
            'url': self.url,
            'source': self.source,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }

class SearchProvider(ABC):
    """Abstract base class for search providers"""
    
    def __init__(self, timeout: int = Config.REQUEST_TIMEOUT):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Maverick-AI-Assistant/2.1'
        })
    
    @abstractmethod
    async def search_async(self, query: str, max_results: int = Config.MAX_SEARCH_RESULTS) -> List[SearchResult]:
        """Async search method"""
        pass
    
    def search(self, query: str, max_results: int = Config.MAX_SEARCH_RESULTS) -> List[SearchResult]:
        """Sync search method"""
        return asyncio.run(self.search_async(query, max_results))
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the search provider is available"""
        pass
    
    def _make_request(self, url: str, params: Dict = None, headers: Dict = None) -> requests.Response:
        """Make HTTP request with error handling"""
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers or {},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {self.__class__.__name__}: {e}")
            raise

class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider"""
    
    async def search_async(self, query: str, max_results: int = Config.MAX_SEARCH_RESULTS) -> List[SearchResult]:
        """Search using DuckDuckGo API"""
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1
        }
        
        try:
            # Run blocking request in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self._make_request(Config.DDG_BASE_URL, params=params)
            )
            data = response.json()
            return self._parse_results(data, max_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict, max_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo response"""
        results = []
        
        # Instant Answer
        if data.get('Answer'):
            results.append(SearchResult(
                title="Instant Answer",
                snippet=data['Answer'],
                url=data.get('AbstractURL', ''),
                source="DuckDuckGo Instant Answer",
                confidence=0.95
            ))
        
        # Abstract
        if data.get('AbstractText') and len(results) < max_results:
            results.append(SearchResult(
                title=data.get('AbstractText', '')[:100],
                snippet=data.get('AbstractText', ''),
                url=data.get('AbstractURL', ''),
                source="DuckDuckGo Abstract",
                confidence=0.85
            ))
        
        # Related Topics
        for topic in data.get('RelatedTopics', []):
            if len(results) >= max_results:
                break
            if isinstance(topic, dict) and topic.get('Text'):
                results.append(SearchResult(
                    title=topic['Text'][:80],
                    snippet=topic['Text'],
                    url=topic.get('FirstURL', ''),
                    source="DuckDuckGo Related",
                    confidence=0.7
                ))
        
        return results[:max_results]
    
    def is_available(self) -> bool:
        """Check DuckDuckGo availability"""
        try:
            response = self._make_request(Config.DDG_BASE_URL, params={"q": "test", "format": "json"})
            return response.status_code == 200
        except:
            return False

class SerperProvider(SearchProvider):
    """Serper Google Search API provider"""
    
    def __init__(self, api_key: str, timeout: int = Config.REQUEST_TIMEOUT):
        super().__init__(timeout)
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Serper API key is required")
    
    async def search_async(self, query: str, max_results: int = Config.MAX_SEARCH_RESULTS) -> List[SearchResult]:
        """Search using Serper API"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        params = {
            'q': query,
            'num': max_results
        }
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._make_request(Config.SERPER_BASE_URL, params=params, headers=headers)
            )
            data = response.json()
            return self._parse_results(data, max_results)
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict, max_results: int) -> List[SearchResult]:
        """Parse Serper response"""
        results = []
        
        # Knowledge Graph
        if data.get('knowledgeGraph'):
            kg = data['knowledgeGraph']
            results.append(SearchResult(
                title=kg.get('title', ''),
                snippet=kg.get('description', ''),
                url=kg.get('website', ''),
                source="Google Knowledge Graph",
                confidence=0.95
            ))
        
        # Organic Results
        for result in data.get('organic', []):
            if len(results) >= max_results:
                break
            results.append(SearchResult(
                title=result.get('title', ''),
                snippet=result.get('snippet', ''),
                url=result.get('link', ''),
                source="Google Search",
                confidence=0.8
            ))
        
        return results[:max_results]
    
    def is_available(self) -> bool:
        """Check Serper availability"""
        if not self.api_key:
            return False
        try:
            headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
            response = self._make_request(Config.SERPER_BASE_URL, 
                                        params={'q': 'test'}, headers=headers)
            return response.status_code == 200
        except:
            return False

class ScrapingDogProvider(SearchProvider):
    """ScrapingDog Google Search API provider"""
    
    def __init__(self, api_key: str, timeout: int = Config.REQUEST_TIMEOUT):
        super().__init__(timeout)
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("ScrapingDog API key is required")
    
    async def search_async(self, query: str, max_results: int = Config.MAX_SEARCH_RESULTS) -> List[SearchResult]:
        """Search using ScrapingDog API"""
        params = {
            "api_key": self.api_key,
            "query": query,
            "results": min(max_results, 10),
            "country": "US",
            "page": 0
        }
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._make_request(Config.SCRAPINGDOG_BASE_URL, params=params)
            )
            data = response.json()
            return self._parse_results(data, max_results)
        except Exception as e:
            logger.error(f"ScrapingDog search failed: {e}")
            return []
    
    def _parse_results(self, data: Dict, max_results: int) -> List[SearchResult]:
        """Parse ScrapingDog response"""
        results = []
        
        for result in data.get('organic_results', []):
            if len(results) >= max_results:
                break
            results.append(SearchResult(
                title=result.get('title', ''),
                snippet=result.get('snippet', ''),
                url=result.get('link', ''),
                source="ScrapingDog Search",
                confidence=0.75
            ))
        
        return results[:max_results]
    
    def is_available(self) -> bool:
        """Check ScrapingDog availability"""
        if not self.api_key:
            return False
        try:
            params = {"api_key": self.api_key, "query": "test", "results": 1}
            response = self._make_request(Config.SCRAPINGDOG_BASE_URL, params=params)
            return response.status_code == 200
        except:
            return False

class WebSearchTool(BaseTool):
    """LangChain tool for web search with multiple providers"""
    
    name: str = "web_search"
    description: str = "Search the web for current information when you don't know something or need recent data"
    providers: List[SearchProvider] = Field(default_factory=list)
    
    def __init__(self):
        super().__init__()
        self.providers = self._initialize_providers()
        logger.info(f"Initialized {len(self.providers)} search providers")
    
    def _initialize_providers(self) -> List[SearchProvider]:
        """Initialize available search providers"""
        providers = []
        
        # DuckDuckGo (always available)
        ddg_provider = DuckDuckGoProvider()
        if ddg_provider.is_available():
            providers.append(ddg_provider)
            logger.info("DuckDuckGo provider initialized")
        
        # Serper (if API key available)
        if Config.SERPER_API_KEY:
            try:
                serper_provider = SerperProvider(Config.SERPER_API_KEY)
                if serper_provider.is_available():
                    providers.append(serper_provider)
                    logger.info("Serper provider initialized")
            except ValueError as e:
                logger.warning(f"Serper provider initialization failed: {e}")
        
        # ScrapingDog (if API key available)
        if Config.SCRAPINGDOG_API_KEY:
            try:
                scrapingdog_provider = ScrapingDogProvider(Config.SCRAPINGDOG_API_KEY)
                if scrapingdog_provider.is_available():
                    providers.append(scrapingdog_provider)
                    logger.info("ScrapingDog provider initialized")
            except ValueError as e:
                logger.warning(f"ScrapingDog provider initialization failed: {e}")
        
        return providers
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute web search"""
        logger.info(f"Searching for: {query}")
        
        
        # Try each provider in order (DuckDuckGo first)
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            try:
                if not provider.is_available():
                    logger.info(f"{provider_name} not available, skipping")
                    continue
                
                logger.info(f"Trying {provider_name}")
                results = provider.search(query, Config.MAX_SEARCH_RESULTS)
                if results:
                    logger.info(f"Found {len(results)} results using {provider_name}")
                    return self._format_results(results)
                else:
                    logger.info(f"No results from {provider_name}, trying next provider")
                
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                continue
        
        return "No search results found from any provider."
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Async version of search"""
        logger.info(f"Async searching for: {query}")
        
        # # Add location prefix
        # localized_query = f"location: Pakistan {query}"
        # logger.info(f"Localized query: {localized_query}")
        
        # Try each provider in order (DuckDuckGo first)
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            try:
                if not provider.is_available():
                    logger.info(f"{provider_name} not available, skipping")
                    continue
                
                logger.info(f"Trying {provider_name}")
                results = await provider.search_async(query, Config.MAX_SEARCH_RESULTS)
                if results:
                    logger.info(f"Found {len(results)} results using {provider_name}")
                    return self._format_results(results)
                
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                continue
        
        return "No search results found from any provider."
    
    def _format_results(self, results: List[SearchResult]) -> str:
        """Format search results for LLM consumption"""
        if not results:
            return "No results found."
        
        formatted = ["Search Results:"]
        for i, result in enumerate(results[:3], 1):  # Limit to top 3 for brevity
            formatted.append(f"\n{i}. {result.title}")
            if result.snippet:
                # Truncate snippet for chat-like brevity
                snippet = result.snippet[:200] + "..." if len(result.snippet) > 200 else result.snippet
                formatted.append(f"   {snippet}")
            if result.url:
                formatted.append(f"   Source: {result.source}")
        
        return "\n".join(formatted)

class GroqLLM(LLM):
    """Custom LangChain LLM for Groq API with structured output support"""
    
    model_name: str = Config.DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 300
    api_key: str = Field(default_factory=lambda: Config.GROQ_API_KEY or "")
    headers: Dict[str, str] = Field(default_factory=dict)
    
    def __init__(self, **kwargs):
        """Initialize the LLM"""
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Make API call to Groq"""
        
        # Convert prompt to messages format
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.post(
                    f"{Config.GROQ_BASE_URL}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=Config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
                
            except Exception as e:
                logger.warning(f"Groq API attempt {attempt + 1} failed: {e}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY * (2 ** attempt))
                else:
                    return "I'm having trouble processing that right now. Please try again! 🤔"
    
    def get_search_decision(self, message: str, conversation_context: str = "") -> SearchDecision:
        """Use LLM to decide if search is needed with structured output"""
        
        decision_prompt = f"""Analyze this user message and decide if web search is needed.

User message: "{message}"

Recent conversation context: {conversation_context}

Consider these factors:
- Does the question require current/recent information?
- Is it about events, news, prices, weather, or time-sensitive data?
- Are there specific people, companies, or entities that might have recent updates?
- Is it asking about current status, latest versions, or recent developments?
- Would my knowledge cutoff affect the accuracy of my response?

Respond with a JSON object matching this format:
{{
    "should_search": true/false,
    "search_query": "optimized search query or null",
    "reasoning": "brief explanation of decision",
    "confidence": 0.0-1.0
}}

Be conservative - only search when truly necessary for accuracy."""

        try:
            response = self._call(decision_prompt)
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                decision_data = json.loads(json_str)
                return SearchDecision(**decision_data)
            else:
                # Fallback to simple heuristics
                return self._fallback_search_decision(message)
                
        except Exception as e:
            logger.warning(f"Structured search decision failed: {e}")
            return self._fallback_search_decision(message)
    
    def _fallback_search_decision(self, message: str) -> SearchDecision:
        """Fallback search decision using simple heuristics"""
        message_lower = message.lower()
        
        # Simple keyword-based detection
        time_sensitive_keywords = [
            'latest', 'recent', 'current', 'today', 'now', 'news', 'update',
            'price', 'stock', 'weather', 'who is', 'what happened', 'when did'
        ]
        
        should_search = any(keyword in message_lower for keyword in time_sensitive_keywords)
        
        return SearchDecision(
            should_search=should_search,
            search_query=message if should_search else None,
            reasoning="Fallback keyword-based detection",
            confidence=0.6 if should_search else 0.8
        )

class Maverick:
    """
    Maverick - Professional AI Assistant with Advanced Web Search
    
    Features:
    - LangChain-powered conversation management
    - Multi-provider web search with intelligent fallbacks
    - LLM-powered search decision making
    - Professional error handling and logging
    """
    
    def __init__(self, model: str = Config.DEFAULT_MODEL):
        """Initialize Maverick with LangChain components"""
        
        # Initialize LLM
        self.llm = GroqLLM(model_name=model)
        
        # Initialize web search tool
        self.search_tool = WebSearchTool()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=Config.MAX_CONVERSATION_HISTORY,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # System prompt for Maverick's personality
        self.system_prompt = """You are Maverick, 

Key characteristics:
- Professional yet approachable, like chatting with a knowledgeable friend
- Keep responses concise and engaging (3-5 sentences typically)
- Use appropriate emojis sparingly for warmth
- Be direct and helpful without being overly formal
- When you don't know current information, use the web search tool
- Synthesize information naturally into conversational responses
- Admit when you're uncertain rather than guessing

You have access to real-time web search. The system will automatically determine when to search based on:
- Recent events, news, or current data needs
- Questions about prices, stocks, weather, or time-sensitive information
- When factual claims need verification
- User requests for "latest" or "current" information

Always respond naturally and conversationally, as if texting a smart friend."""
        
        # Create conversation chain
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        logger.info("✅ Maverick initialized successfully")
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation context for search decisions"""
        messages = self.memory.chat_memory.messages[-6:]  # Last 3 exchanges
        context_parts = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts)
    
    async def chat_async(self, message: str) -> str:
        """Async chat method"""
        if not message.strip():
            return "Hey there! What's on your mind? 😊"
        
        try:
            # Get conversation context memory_window
            context = self._get_conversation_context()
            
            # Let LLM decide if search is needed
            search_decision = self.llm.get_search_decision(message, context)
            
            if search_decision.should_search and search_decision.confidence > 0.5:
                logger.info(f"🔍 Search triggered: {search_decision.reasoning}")
                search_query = search_decision.search_query or message
                search_results = await self.search_tool._arun(search_query)
                
                # Create enhanced prompt with search results
                enhanced_message = f"""User query: {message}

{search_results}

Please provide a natural, conversational response incorporating this information."""
                
                # Generate response with search context
                response = self.llm._call(
                    prompt=self._build_conversation_prompt(enhanced_message)
                )
            else:
                # Regular conversation without search
                response = self.llm._call(
                    prompt=self._build_conversation_prompt(message)
                )
            
            # Update memory
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "Oops! Something went wrong on my end. Mind trying that again? 🤖"
    
    def chat(self, message: str) -> str:
        """Sync chat method"""
        return asyncio.run(self.chat_async(message))
    
    def _build_conversation_prompt(self, message: str) -> str:
        """Build prompt with conversation history"""
        # Get chat history
        history = self.memory.chat_memory.messages
        
        # Build full prompt
        full_prompt = f"{self.system_prompt}\n\n"
        
        # Add conversation history
        for msg in history[-10:]:  # Last 10 messages
            if isinstance(msg, HumanMessage):
                full_prompt += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                full_prompt += f"Assistant: {msg.content}\n"
        
        # Add current message
        full_prompt += f"Human: {message}\nAssistant:"
        
        return full_prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        messages = self.memory.chat_memory.messages
        return {
            "total_messages": len(messages),
            "available_search_providers": len(self.search_tool.providers),
            "model": self.llm.model_name,
            "memory_window": Config.MAX_CONVERSATION_HISTORY
        }
    
    def clear_conversation(self) -> str:
        """Clear conversation history"""
        self.memory.clear()
        logger.info("Conversation history cleared")
        return "Fresh start! What would you like to chat about? 🆕"
    
    def change_model(self, model: str) -> str:
        """Change the underlying LLM model"""
        old_model = self.llm.model_name
        self.llm.model_name = model
        logger.info(f"Model changed from {old_model} to {model}")
        return f"✅ Switched to {model}. Ready to chat!"

# Convenience functions
def create_maverick(model: str = Config.DEFAULT_MODEL) -> Maverick:
    """Factory function to create Maverick instance"""
    return Maverick(model=model)

def run_console_chat():
    """Run console-based chat interface"""
    try:
        print("🚀 Starting Maverick - Professional AI Assistant")
        print("=" * 50)
        
        maverick = create_maverick()
        print("✅ Maverick is ready! Type 'quit' to exit, 'clear' to reset, 'stats' for info.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Maverick: Take care! Chat with you soon! 👋")
                    break
                elif user_input.lower() == 'clear':
                    response = maverick.clear_conversation()
                    print(f"Maverick: {response}")
                    continue
                elif user_input.lower() == 'stats':
                    stats = maverick.get_stats()
                    print(f"Maverick: Here are my stats: {json.dumps(stats, indent=2)}")
                    continue
                elif not user_input:
                    continue
                
                print("Maverick: ", end="", flush=True)
                response = maverick.chat(user_input)
                print(f"{response}\n")
                
            except KeyboardInterrupt:
                print("\n\nMaverick: Catch you later! 👋")
                break
                
    except Exception as e:
        print(f"❌ Error starting Maverick: {e}")
        print("\nPlease ensure you have:")
        print("- GROQ_API_KEY set in your environment")
        print("- Optional: SERPER_API_KEY, SCRAPINGDOG_API_KEY for enhanced search")

if __name__ == "__main__":
    run_console_chat()