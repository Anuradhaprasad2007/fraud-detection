from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from src.database import Base


class TransactionRecord(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    step = Column(Integer)
    type = Column(String)
    amount = Column(Float)
    oldbalanceOrg = Column(Float)
    newbalanceOrig = Column(Float)
    oldbalanceDest = Column(Float)
    newbalanceDest = Column(Float)
    fraud_probability = Column(Float)
    is_fraud = Column(Boolean)
    created_at = Column(DateTime(timezone=True), server_default=func.now())