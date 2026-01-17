"""
Gemini LLM - Generate contextual answers using Google's Gemini Pro
Implements RAG response generation with customizable prompts
"""

from typing import Optional, List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()


class GeminiLLM:
    """
    Language Model integration using Google's Gemini Pro.
    Generates answers based on retrieved context.
    """
    
    # Model configuration
    MODEL_NAME = "models/gemini-2.5-flash"
    
    # Default system prompt for the FAQ bot
    DEFAULT_SYSTEM_PROMPT = """You are ThriveBot, a helpful AI assistant for Thrive Scholars. 
Your role is to answer questions accurately using the provided context from our knowledge base.

Guidelines:
1. Base your answers ONLY on the provided context. Do not make up information.
2. If the context doesn't contain enough information to answer, say so honestly.
3. Be concise but thorough. Use bullet points for clarity when appropriate.
4. Be warm, encouraging, and supportive - you're helping scholars succeed!
5. If asked about deadlines or specific dates, always recommend verifying with official sources.
6. For sensitive topics (financial aid, personal issues), recommend speaking with a Thrive advisor.

Remember: You represent Thrive Scholars, so be professional, inclusive, and helpful."""

    def __init__(
        self,
        api_key: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        self.api_key = api_key
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = None
        self._configure_model()
    
    def _configure_model(self):
        """Configure the Gemini model"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            
            # Configure generation settings
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.95,
                "top_k": 40
            }
            
            # Configure safety settings (adjust as needed)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            self._model = genai.GenerativeModel(
                model_name=self.MODEL_NAME,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=self.system_prompt
            )
            
            logger.info("Gemini LLM configured", model=self.MODEL_NAME)
            
        except Exception as e:
            logger.error("Failed to configure Gemini LLM", error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response based on query and context.
        
        Args:
            query: User's question
            context: Retrieved context from knowledge base
            conversation_history: Optional list of previous messages
            
        Returns:
            Generated response string
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(query, context, conversation_history)
            
            # Generate response
            response = self._model.generate_content(prompt)
            
            # Extract text from response
            if response.parts:
                answer = response.text
                logger.info(
                    "Generated response",
                    query=query[:50],
                    response_length=len(answer)
                )
                return answer
            else:
                logger.warning("Empty response from LLM", query=query[:50])
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error("Generation failed", error=str(e), query=query[:50])
            return f"I encountered an error while processing your question. Please try again later."
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build the complete prompt for the LLM"""
        
        prompt_parts = []
        
        # Add context
        prompt_parts.append("## Knowledge Base Context")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append("## Previous Conversation")
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("")
        
        # Add current query
        prompt_parts.append("## Current Question")
        prompt_parts.append(query)
        prompt_parts.append("")
        prompt_parts.append("## Your Response")
        prompt_parts.append("Please provide a helpful answer based on the context above:")
        
        return "\n".join(prompt_parts)
    
    async def generate_async(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Async version of generate for better performance"""
        try:
            prompt = self._build_prompt(query, context, conversation_history)
            response = await self._model.generate_content_async(prompt)
            
            if response.parts:
                return response.text
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error("Async generation failed", error=str(e))
            return "I encountered an error while processing your question. Please try again later."
    
    def generate_with_sources(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate response with source citations.
        
        Returns:
            Dict with 'answer' and 'sources' keys
        """
        answer = self.generate(query, context)
        
        # Format sources for display
        formatted_sources = []
        for source in sources[:3]:  # Top 3 sources
            formatted_sources.append({
                "source": source.get("source", "Unknown"),
                "relevance": f"{source.get('score', 0):.0%}"
            })
        
        return {
            "answer": answer,
            "sources": formatted_sources
        }
