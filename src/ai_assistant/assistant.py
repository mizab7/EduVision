"""
AI Teaching Assistant for EduVision AI.
Provides intelligent pedagogical recommendations to teachers using
Google Gemini API (with seamless offline heuristic engine fallback).
"""

import json
import httpx
from typing import Dict, Any, List, Optional
from config.settings import get_settings


class TeachingAssistant:
    """
    Evaluates classroom engagement state and generates actionable pedagogical interventions
    via Google Gemini 1.5 Flash (or local offline pedagogical expert engine).
    """

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.GEMINI_API_KEY
        # If API key is empty or placeholder, mark as offline mode
        if not self.api_key or "your-google-gemini" in self.api_key or "placeholder" in self.api_key.lower():
            self.api_key = None

    def is_gemini_active(self) -> bool:
        """Returns True if a real Gemini API key is configured."""
        return bool(self.api_key)

    def generate_recommendations(
        self,
        class_engagement_index: float,
        students: List[Dict[str, Any]],
        active_alerts: List[str] = None,
        subject: str = "Computer Science / Engineering Lecture",
        duration_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Generates teaching interventions based on real-time classroom telemetry.
        Tries Gemini API first, falls back gracefully to offline pedagogical rules.
        """
        if active_alerts is None:
            active_alerts = []

        # Aggregate statistics
        total_students = len(students)
        drowsy_count = sum(1 for s in students if s.get("is_drowsy"))
        phone_count = sum(1 for s in students if s.get("phone_detected"))
        looking_away = sum(1 for s in students if s.get("gaze") != "CENTER")
        avg_score = class_engagement_index

        context = {
            "subject": subject,
            "duration_minutes": duration_minutes,
            "class_engagement_index": round(avg_score, 1),
            "total_students": total_students,
            "drowsy_count": drowsy_count,
            "phone_count": phone_count,
            "looking_away_count": looking_away,
            "active_alerts": active_alerts
        }

        # Try Google Gemini if configured
        if self.api_key:
            try:
                gemini_result = self._call_gemini_api(context)
                if gemini_result:
                    return gemini_result
            except Exception as e:
                print(f"Warning: Gemini API call failed ({e}), switching to offline engine.")

        # Default to offline pedagogical engine
        return self._generate_offline_recommendations(context)

    def _call_gemini_api(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Invokes Google Gemini 1.5 Flash via REST API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"

        prompt = f"""
You are EduVision AI's pedagogical Teaching Assistant for university-level engineering classes.
Analyze the following real-time classroom computer vision engagement telemetry:
- Subject: {ctx['subject']}
- Elapsed Time: {ctx['duration_minutes']} minutes
- Class Engagement Index (CEI): {ctx['class_engagement_index']}%
- Total Students Monitored: {ctx['total_students']}
- Drowsy / Micro-sleep: {ctx['drowsy_count']} students
- Active Phone Distractions: {ctx['phone_count']} students
- Looking Away / Disengaged Posture: {ctx['looking_away_count']} students
- Active Alerts: {', '.join(ctx['active_alerts']) if ctx['active_alerts'] else 'None'}

Provide an immediate pedagogical intervention plan in valid JSON format with this structure:
{{
  "summary": "1-2 sentence assessment of classroom attention flow",
  "urgency": "High | Medium | Low | Optimal",
  "immediate_action": {{
    "title": "Short title of 30-second action",
    "rationale": "Why this works for current attention state",
    "steps": ["Step 1", "Step 2"]
  }},
  "interactive_exercise": {{
    "title": "Short title for 2-3 minute engagement exercise",
    "description": "How to execute it in the classroom"
  }},
  "pacing_advice": "Advice on tone, slide progression, or break timing"
}}
Return ONLY valid JSON.
"""

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json"
            }
        }

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text)
                parsed["source"] = "Gemini 1.5 Flash"
                parsed["context"] = ctx
                return parsed
        return None

    def _generate_offline_recommendations(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expert pedagogical rule-based system providing immediate, evidence-backed
        teaching interventions for university and engineering educators.
        """
        cei = ctx["class_engagement_index"]
        phone_count = ctx["phone_count"]
        drowsy_count = ctx["drowsy_count"]
        looking_away = ctx["looking_away_count"]

        # Case 1: Critical disengagement (CEI < 45% or high fatigue)
        if cei < 45.0 or drowsy_count >= 2:
            return {
                "source": "EduVision Offline Pedagogical Engine",
                "urgency": "High",
                "summary": (
                    f"Classroom attention has dropped to {cei:.1f}% with noticeable fatigue indicators. "
                    "A rapid cognitive pattern interrupt is needed to regain focus."
                ),
                "immediate_action": {
                    "title": "30-Second Cognitive Reset & Stretch",
                    "rationale": (
                        "Physical posture change stimulates blood flow to the prefrontal cortex, "
                        "clearing afternoon cognitive sluggishness."
                    ),
                    "steps": [
                        "Pause your slide immediately and ask everyone to stand up for a 30-second shoulder roll.",
                        "Ask students to quickly look at the back wall to reset visual accommodation from the screen."
                    ]
                },
                "interactive_exercise": {
                    "title": "Think-Pair-Share on a Controversial Problem",
                    "description": (
                        "Pose a challenging conceptual bug or design trade-off. Give students 60 seconds to "
                        "discuss with their desk partner before cold-calling two pairs for quick opinions."
                    )
                },
                "pacing_advice": (
                    "Decrease continuous monologue duration. Break slides every 8–10 minutes with "
                    "a live code demo or hand-drawn board architecture."
                ),
                "context": ctx
            }

        # Case 2: Active phone distractions
        elif phone_count > 0:
            return {
                "source": "EduVision Offline Pedagogical Engine",
                "urgency": "Medium",
                "summary": (
                    f"{phone_count} student(s) detected using mobile devices. "
                    "Convert device distraction into active learning engagement."
                ),
                "immediate_action": {
                    "title": "BYOD Real-Time Concept Poll",
                    "rationale": (
                        "Rather than policing devices, directing phone usage towards an immediate lecture poll "
                        "channels divided attention back into topic comprehension."
                    ),
                    "steps": [
                        "Launch a quick 1-question poll (e.g., Slido, Kahoot, or Google Form) on the screen.",
                        "Give 60 seconds to vote on the next design choice in today's topic."
                    ]
                },
                "interactive_exercise": {
                    "title": "Spot-the-Bug Live Challenge",
                    "description": (
                        "Project 10 lines of code containing a subtle runtime error and challenge students "
                        "to be the first to call out the line number."
                    )
                },
                "pacing_advice": (
                    "Walk between student aisles while explaining concepts to reduce blind spots "
                    "where off-task screen browsing typically occurs."
                ),
                "context": ctx
            }

        # Case 3: Moderate disengagement / wandering attention
        elif cei < 70.0 or looking_away > 0:
            return {
                "source": "EduVision Offline Pedagogical Engine",
                "urgency": "Medium",
                "summary": (
                    f"Class attention is moderate ({cei:.1f}%). Gaze patterns suggest students are losing "
                    "the thread of abstract conceptual delivery."
                ),
                "immediate_action": {
                    "title": "Socratic Checkpoint Question",
                    "rationale": (
                        "Direct interactive questioning pulls wandering attention back to the screen."
                    ),
                    "steps": [
                        "Stop speaking for 3 seconds to create an intentional silence cue.",
                        "Ask: 'If we inverted this condition, what happens to memory complexity?'"
                    ]
                },
                "interactive_exercise": {
                    "title": "Finger Voting / Quick Hand Show",
                    "description": (
                        "Offer 3 options (A, B, C) and ask everyone to hold up 1, 2, or 3 fingers simultaneously."
                    )
                },
                "pacing_advice": (
                    "Increase voice modulation and vary pitch. Walk closer to the display screen to point out "
                    "specific data flow arrows."
                ),
                "context": ctx
            }

        # Case 4: High engagement / Flow state (CEI >= 70%)
        else:
            return {
                "source": "EduVision Offline Pedagogical Engine",
                "urgency": "Optimal",
                "summary": (
                    f"Excellent class engagement ({cei:.1f}%). Students are focused with forward gaze and active posture."
                ),
                "immediate_action": {
                    "title": "Maintain Rhythm & Reinforce Understanding",
                    "rationale": "Class is in optimal cognitive flow. Maintain momentum without disruptive pauses.",
                    "steps": [
                        "Acknowledge student focus: 'Great attention here, this next part is key.'",
                        "Continue directly into the core implementation demonstration."
                    ]
                },
                "interactive_exercise": {
                    "title": "Advanced Edge-Case Challenge",
                    "description": (
                        "Present an industry-grade edge case to push high-performing attention towards critical analysis."
                    )
                },
                "pacing_advice": (
                    "Keep your current cadence. Ensure key terms are highlighted visually on the slides."
                ),
                "context": ctx
            }
