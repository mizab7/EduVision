from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, LargeBinary
from sqlalchemy.orm import relationship
import datetime

from src.database.connection import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    roll_number = Column(String, unique=True, index=True)
    department = Column(String)
    semester = Column(Integer)
    enrolled_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    embedding_data = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String)
    faculty_name = Column(String)
    classroom = Column(String)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    class_engagement_index = Column(Float, nullable=True)


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    session_id = Column(Integer, ForeignKey("sessions.id"))
    marked_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_spoofed = Column(Boolean, default=False)
    confidence = Column(Float)


class EngagementLog(Base):
    __tablename__ = "engagement_logs"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    session_id = Column(Integer, ForeignKey("sessions.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    attentiveness_score = Column(Float)
    gaze_score = Column(Float)
    head_pose_score = Column(Float)
    blink_rate = Column(Float)
    is_yawning = Column(Boolean)
    is_drowsy = Column(Boolean)
    phone_detected = Column(Boolean)


class SpoofAttempt(Base):
    __tablename__ = "spoof_attempts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    spoof_type = Column(String)
    image_path = Column(String, nullable=True)


class AlertLog(Base):
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    alert_type = Column(String)
    message = Column(String)
    recommendation = Column(String)
    acknowledged = Column(Boolean, default=False)
