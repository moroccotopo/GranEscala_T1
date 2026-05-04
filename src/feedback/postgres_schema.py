import os
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
DATABASE_URL = os.getenv("DATABASE_URL", "")


class BusinessFeedback(Base):
    __tablename__ = "business_feedback"

    feedback_id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(200))
    item_id = Column(BigInteger)
    shop_id = Column(BigInteger)
    item_category_id = Column(BigInteger)
    severity = Column(String(50))
    comment = Column(Text)
    status = Column(String(50), default="open")


def engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no configurado")
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def init_db():
    Base.metadata.create_all(engine())


def add_feedback(created_by, item_id, shop_id, item_category_id, severity, comment, status="open", created_at=None):
    init_db()
    Session = sessionmaker(bind=engine())
    with Session() as session:
        obj = BusinessFeedback(
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(timezone.utc),
            created_by=created_by,
            item_id=item_id,
            shop_id=shop_id,
            item_category_id=item_category_id,
            severity=severity,
            comment=comment,
            status=status,
        )
        session.add(obj)
        session.commit()
        return obj.feedback_id


def list_feedback(limit=100):
    init_db()
    Session = sessionmaker(bind=engine())
    with Session() as session:
        rows = session.execute(
            select(BusinessFeedback).order_by(BusinessFeedback.created_at.desc()).limit(limit)
        ).scalars().all()
        return [
            {
                "feedback_id": r.feedback_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "created_by": r.created_by,
                "item_id": r.item_id,
                "shop_id": r.shop_id,
                "item_category_id": r.item_category_id,
                "severity": r.severity,
                "comment": r.comment,
                "status": r.status,
            }
            for r in rows
        ]


if __name__ == "__main__":
    init_db()
    print("Tablas de feedback listas")
