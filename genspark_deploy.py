"""
SolarVisionAI - GenSpark AI Agent Deployment
Integration with GenSpark AI platform for intelligent defect analysis

Features:
- API wrapper for GenSpark platform
- Agent configuration and deployment
- Webhook handlers for real-time updates
- Multi-platform support
- Intelligent query routing
- Context-aware responses

Author: SolarVisionAI Team
Version: 1.0.0
"""

import requests
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import hashlib
import hmac

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """AI agent capabilities"""
    DEFECT_ANALYSIS = "defect_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"
    RECOMMENDATION = "recommendation"
    COMPLIANCE_CHECK = "compliance_check"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class GenSparkConfig:
    """GenSpark platform configuration"""
    api_key: str
    api_endpoint: str = "https://api.genspark.ai/v1"
    agent_id: Optional[str] = None
    webhook_secret: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_streaming: bool = False


@dataclass
class AgentMessage:
    """Message for AI agent"""
    message_id: str
    content: str
    role: str = "user"  # user, assistant, system
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'message_id': self.message_id,
            'content': self.content,
            'role': self.role,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'attachments': self.attachments
        }


@dataclass
class AgentResponse:
    """Response from AI agent"""
    response_id: str
    content: str
    confidence: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'response_id': self.response_id,
            'content': self.content,
            'confidence': self.confidence,
            'suggestions': self.suggestions,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms
        }


class GenSparkAgent:
    """
    GenSpark AI agent client

    Provides interface to GenSpark AI platform for:
    - Defect analysis interpretation
    - Quality assessment guidance
    - Compliance recommendations
    - Interactive troubleshooting
    """

    def __init__(self, config: GenSparkConfig):
        """
        Initialize GenSpark agent

        Args:
            config: GenSpark configuration
        """
        self.config = config
        self.conversation_history: List[AgentMessage] = []
        self.session_id = self._generate_session_id()

        logger.info(f"Initialized GenSparkAgent with session: {self.session_id}")

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]

    def send_message(
        self,
        message: str,
        context: Optional[Dict] = None,
        attachments: Optional[List[str]] = None
    ) -> AgentResponse:
        """
        Send message to AI agent

        Args:
            message: User message
            context: Optional context dictionary
            attachments: Optional file attachments

        Returns:
            AgentResponse
        """
        import time
        start_time = time.time()

        # Create message object
        agent_message = AgentMessage(
            message_id=f"{self.session_id}_{len(self.conversation_history)}",
            content=message,
            role="user",
            metadata=context or {},
            attachments=attachments or []
        )

        self.conversation_history.append(agent_message)

        try:
            # Prepare request payload
            payload = {
                'session_id': self.session_id,
                'message': message,
                'context': context or {},
                'conversation_history': [
                    msg.to_dict() for msg in self.conversation_history[-5:]  # Last 5 messages
                ]
            }

            # Add agent ID if available
            if self.config.agent_id:
                payload['agent_id'] = self.config.agent_id

            # Make API request
            headers = {
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json'
            }

            response = requests.post(
                f"{self.config.api_endpoint}/chat",
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds
            )

            response.raise_for_status()
            result = response.json()

            # Create response object
            agent_response = AgentResponse(
                response_id=result.get('response_id', f"resp_{len(self.conversation_history)}"),
                content=result.get('content', result.get('message', '')),
                confidence=result.get('confidence', 0.0),
                suggestions=result.get('suggestions', []),
                metadata=result.get('metadata', {}),
                processing_time_ms=(time.time() - start_time) * 1000
            )

            # Add assistant response to history
            assistant_message = AgentMessage(
                message_id=agent_response.response_id,
                content=agent_response.content,
                role="assistant"
            )
            self.conversation_history.append(assistant_message)

            logger.info(f"Agent response received in {agent_response.processing_time_ms:.0f}ms")

            return agent_response

        except requests.exceptions.RequestException as e:
            logger.error(f"GenSpark API error: {str(e)}")

            # Return fallback response
            return AgentResponse(
                response_id="error",
                content=f"Sorry, I encountered an error: {str(e)}",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def analyze_defects(
        self,
        defect_data: Dict,
        include_recommendations: bool = True
    ) -> str:
        """
        Get AI analysis of defect data

        Args:
            defect_data: Defect analysis data
            include_recommendations: Include recommendations

        Returns:
            Analysis text
        """
        # Format defect data for AI
        summary = f"Detected {defect_data.get('total_defects', 0)} defects:\n"

        if 'defects' in defect_data:
            for defect in defect_data['defects'][:5]:  # Top 5
                summary += f"- {defect.get('type', 'Unknown')}: "
                summary += f"Severity {defect.get('severity', 'N/A')}, "
                summary += f"Power loss {defect.get('power_loss_w', 0):.1f}W\n"

        query = f"{summary}\n\n"
        if include_recommendations:
            query += "Please analyze these defects and provide recommendations for:"
            query += "\n1. Immediate actions needed"
            query += "\n2. Long-term preventive measures"
            query += "\n3. Expected impact on module performance"
        else:
            query += "Please provide a brief analysis of these defects."

        response = self.send_message(query, context={'defect_data': defect_data})

        return response.content

    def check_compliance(
        self,
        quality_metrics: Dict,
        standard: str = "IEC 60904-13"
    ) -> str:
        """
        Check compliance with standards

        Args:
            quality_metrics: Quality metrics dictionary
            standard: Standard name

        Returns:
            Compliance analysis
        """
        query = (
            f"Please review these image quality metrics against {standard}:\n\n"
            f"Quality Score: {quality_metrics.get('quality_score', 0)}/100\n"
            f"IEC Sharpness Class: {quality_metrics.get('iec_sharpness_class', 'N/A')}\n"
            f"SNR: {quality_metrics.get('snr_db', 0):.1f} dB\n"
            f"MTF50: {quality_metrics.get('mtf50', 0):.3f}\n\n"
            f"Is this compliant? What improvements are needed?"
        )

        response = self.send_message(query, context={'quality_metrics': quality_metrics})

        return response.content

    def troubleshoot(self, issue_description: str) -> str:
        """
        Get troubleshooting help

        Args:
            issue_description: Description of the issue

        Returns:
            Troubleshooting guidance
        """
        query = (
            f"I'm experiencing the following issue with EL imaging:\n\n"
            f"{issue_description}\n\n"
            f"Can you help me troubleshoot this? Please provide step-by-step guidance."
        )

        response = self.send_message(query)

        return response.content

    def get_recommendations(
        self,
        camera_settings: Dict,
        test_conditions: Dict
    ) -> List[str]:
        """
        Get camera/test setup recommendations

        Args:
            camera_settings: Current camera settings
            test_conditions: Current test conditions

        Returns:
            List of recommendations
        """
        query = (
            f"Current camera settings:\n"
            f"- Exposure: {camera_settings.get('exposure_time_ms', 0)}ms\n"
            f"- ISO: {camera_settings.get('iso', 0)}\n"
            f"- Aperture: f/{camera_settings.get('aperture_fstop', 0)}\n\n"
            f"Test conditions:\n"
            f"- Current: {test_conditions.get('test_current_a', 0):.3f}A\n"
            f"- Temperature: {test_conditions.get('module_temp_c', 25)}°C\n\n"
            f"What settings would you recommend for optimal EL image quality?"
        )

        response = self.send_message(
            query,
            context={
                'camera_settings': camera_settings,
                'test_conditions': test_conditions
            }
        )

        # Extract recommendations from response
        recommendations = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                recommendations.append(line[1:].strip())
            elif line and line[0].isdigit() and '.' in line[:3]:
                recommendations.append(line.split('.', 1)[1].strip())

        return recommendations if recommendations else [response.content]

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")


class WebhookHandler:
    """
    Webhook handler for GenSpark events

    Handles incoming webhooks from GenSpark platform
    """

    def __init__(self, webhook_secret: str):
        """
        Initialize webhook handler

        Args:
            webhook_secret: Secret for webhook signature verification
        """
        self.webhook_secret = webhook_secret
        self.event_handlers: Dict[str, List[Callable]] = {}

        logger.info("Initialized WebhookHandler")

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify webhook signature

        Args:
            payload: Request payload
            signature: Signature from header

        Returns:
            True if valid
        """
        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register event handler

        Args:
            event_type: Event type (e.g., "message.received")
            handler: Handler function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event: {event_type}")

    def handle_event(self, event_data: Dict) -> None:
        """
        Handle webhook event

        Args:
            event_data: Event data dictionary
        """
        event_type = event_data.get('type', 'unknown')

        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Handler error for {event_type}: {str(e)}")
        else:
            logger.warning(f"No handler registered for event type: {event_type}")


# Convenience functions
def create_defect_analysis_agent(api_key: str) -> GenSparkAgent:
    """Create agent specialized in defect analysis"""
    config = GenSparkConfig(
        api_key=api_key,
        agent_id="defect-analyzer-v1"
    )
    return GenSparkAgent(config)


def create_quality_assessment_agent(api_key: str) -> GenSparkAgent:
    """Create agent specialized in quality assessment"""
    config = GenSparkConfig(
        api_key=api_key,
        agent_id="quality-assessor-v1"
    )
    return GenSparkAgent(config)


if __name__ == "__main__":
    # Example usage
    import os

    # Get API key from environment
    api_key = os.getenv('GENSPARK_API_KEY', 'demo_key')

    # Create agent
    agent = create_defect_analysis_agent(api_key)

    # Example: Analyze defects
    defect_data = {
        'total_defects': 3,
        'defects': [
            {'type': 'crack', 'severity': 'critical', 'power_loss_w': 15.2},
            {'type': 'hotspot', 'severity': 'high', 'power_loss_w': 8.5},
            {'type': 'dead_cell', 'severity': 'medium', 'power_loss_w': 3.2}
        ]
    }

    print("Analyzing defects...")
    analysis = agent.analyze_defects(defect_data)
    print(f"\nAI Analysis:\n{analysis}")

    # Example: Check compliance
    quality_metrics = {
        'quality_score': 75.3,
        'iec_sharpness_class': 'B',
        'snr_db': 28.5,
        'mtf50': 0.52
    }

    print("\n" + "=" * 60)
    print("Checking compliance...")
    compliance = agent.check_compliance(quality_metrics)
    print(f"\nCompliance Check:\n{compliance}")

    # Example: Get recommendations
    camera_settings = {
        'exposure_time_ms': 1000,
        'iso': 400,
        'aperture_fstop': 5.6
    }

    test_conditions = {
        'test_current_a': 0.95,
        'module_temp_c': 25
    }

    print("\n" + "=" * 60)
    print("Getting recommendations...")
    recommendations = agent.get_recommendations(camera_settings, test_conditions)
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
