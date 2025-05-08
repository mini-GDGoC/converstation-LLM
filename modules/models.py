from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MenuItem(Base):
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer)
    name = Column(String, nullable=False)
    description = Column(Text)
    emoji = Column(String)
    keywords = Column(Text)  # JSON 문자열 저장됨
