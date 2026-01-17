"""
Slack Bot - Slack Bolt integration for ThriveBot
Handles Slack events, commands, and message formatting
"""

import asyncio
from typing import Dict, Any, Optional
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
import structlog

from app.config import settings

logger = structlog.get_logger()


class ThriveSlackBot:
    """
    Slack bot using Bolt framework.
    Handles mentions, DMs, and slash commands.
    """
    
    def __init__(
        self,
        retriever,
        llm,
        bot_token: Optional[str] = None,
        app_token: Optional[str] = None,
        signing_secret: Optional[str] = None
    ):
        self.retriever = retriever
        self.llm = llm
        
        # Get tokens from settings or parameters
        self.bot_token = bot_token or settings.slack_bot_token
        self.app_token = app_token or settings.slack_app_token
        self.signing_secret = signing_secret or settings.slack_signing_secret
        
        # Initialize Slack app
        self.app = App(
            token=self.bot_token,
            signing_secret=self.signing_secret
        )
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("ThriveSlackBot initialized")
    
    def _register_handlers(self):
        """Register all Slack event handlers"""
        
        # Handle @mentions
        @self.app.event("app_mention")
        def handle_mention(event, say, client):
            self._process_message(event, say, client)
        
        # Handle direct messages
        @self.app.event("message")
        def handle_dm(event, say, client):
            # Only respond to DMs (not channel messages without mention)
            if event.get("channel_type") == "im":
                self._process_message(event, say, client)
        
        # Slash command for quick queries
        @self.app.command("/thrive")
        def handle_thrive_command(ack, command, respond):
            ack()  # Acknowledge immediately
            self._process_command(command, respond)
        
        # Slash command for help
        @self.app.command("/thrivehelp")
        def handle_help_command(ack, respond):
            ack()
            self._show_help(respond)
    
    def _process_message(self, event: Dict[str, Any], say, client):
        """Process incoming message and generate response"""
        try:
            # Get the message text
            text = event.get("text", "")
            user_id = event.get("user", "unknown")
            channel = event.get("channel")
            
            # Remove bot mention from text
            text = self._clean_message(text)
            
            if not text.strip():
                say("Hi! I'm ThriveBot 👋 Ask me anything about Thrive Scholars!")
                return
            
            # Show typing indicator
            client.chat_postMessage(
                channel=channel,
                text="🤔 Let me look that up for you..."
            )
            
            # Retrieve relevant context
            context, sources = self.retriever.retrieve_and_format(text)
            
            # Generate response
            response = self.llm.generate(text, context)
            
            # Format and send response
            formatted_response = self._format_response(response, sources)
            say(formatted_response)
            
            logger.info(
                "Processed message",
                user=user_id,
                query=text[:50],
                num_sources=len(sources)
            )
            
        except Exception as e:
            logger.error("Error processing message", error=str(e))
            say("I apologize, but I encountered an error. Please try again later. 🙏")
    
    def _process_command(self, command: Dict[str, Any], respond):
        """Process slash command"""
        try:
            text = command.get("text", "")
            user_id = command.get("user_id", "unknown")
            
            if not text.strip():
                respond("Please provide a question after the command. Example: `/thrive What are the scholarship requirements?`")
                return
            
            # Retrieve and generate
            context, sources = self.retriever.retrieve_and_format(text)
            response = self.llm.generate(text, context)
            
            # Send response
            formatted_response = self._format_response(response, sources)
            respond(formatted_response)
            
            logger.info("Processed command", user=user_id, query=text[:50])
            
        except Exception as e:
            logger.error("Error processing command", error=str(e))
            respond("I encountered an error. Please try again later.")
    
    def _show_help(self, respond):
        """Show help message"""
        help_text = """
*ThriveBot Help* 🎓

I'm here to help you with questions about Thrive Scholars! Here's how to use me:

*In any channel:*
• Mention me: `@ThriveBot What are the scholarship deadlines?`

*In direct messages:*
• Just send me a message directly!

*Slash commands:*
• `/thrive [question]` - Ask a quick question
• `/thrivehelp` - Show this help message

*Example questions:*
• "What are the eligibility requirements?"
• "How do I apply for financial aid?"
• "What mentorship programs are available?"
• "When are the application deadlines?"

_I'll search our knowledge base and give you the best answer I can!_
        """
        respond(help_text)
    
    def _clean_message(self, text: str) -> str:
        """Remove bot mention and clean message text"""
        import re
        # Remove @mentions
        text = re.sub(r'<@[A-Z0-9]+>', '', text)
        return text.strip()
    
    def _format_response(
        self,
        response: str,
        sources: list,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Format response with Slack blocks for rich formatting"""
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": response
                }
            }
        ]
        
        # Add sources if available
        if include_sources and sources:
            source_text = "*Sources:* " + ", ".join([
                f"`{s.get('source', 'Unknown').split('/')[-1]}`" 
                for s in sources[:3]
            ])
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": source_text
                    }
                ]
            })
        
        return {"blocks": blocks, "text": response}
    
    def start_socket_mode(self):
        """Start the bot in Socket Mode (for development)"""
        logger.info("Starting ThriveBot in Socket Mode...")
        handler = SocketModeHandler(self.app, self.app_token)
        handler.start()
    
    def get_bolt_app(self) -> App:
        """Return the Bolt app for integration with FastAPI"""
        return self.app


# Async version for better performance
class AsyncThriveSlackBot:
    """Async version of ThriveSlackBot for better performance"""
    
    def __init__(
        self,
        retriever,
        llm,
        bot_token: Optional[str] = None,
        signing_secret: Optional[str] = None
    ):
        self.retriever = retriever
        self.llm = llm
        
        self.bot_token = bot_token or settings.slack_bot_token
        self.signing_secret = signing_secret or settings.slack_signing_secret
        
        self.app = AsyncApp(
            token=self.bot_token,
            signing_secret=self.signing_secret
        )
        
        self._register_handlers()
    
    def _register_handlers(self):
        """Register async event handlers"""
        
        @self.app.event("app_mention")
        async def handle_mention(event, say, client):
            await self._process_message_async(event, say, client)
        
        @self.app.event("message")
        async def handle_dm(event, say, client):
            if event.get("channel_type") == "im":
                await self._process_message_async(event, say, client)
    
    async def _process_message_async(self, event, say, client):
        """Async message processing"""
        try:
            text = event.get("text", "")
            text = self._clean_message(text)
            
            if not text.strip():
                await say("Hi! I'm ThriveBot 👋 Ask me anything about Thrive Scholars!")
                return
            
            # Retrieve and generate
            context, sources = self.retriever.retrieve_and_format(text)
            response = await self.llm.generate_async(text, context)
            
            await say(self._format_response(response, sources))
            
        except Exception as e:
            logger.error("Error in async processing", error=str(e))
            await say("I apologize, but I encountered an error. Please try again later. 🙏")
    
    def _clean_message(self, text: str) -> str:
        import re
        return re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    
    def _format_response(self, response: str, sources: list) -> Dict[str, Any]:
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": response}}]
        if sources:
            source_text = "*Sources:* " + ", ".join([
                f"`{s.get('source', 'Unknown').split('/')[-1]}`" 
                for s in sources[:3]
            ])
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": source_text}]})
        return {"blocks": blocks, "text": response}
    
    def get_bolt_app(self) -> AsyncApp:
        return self.app
