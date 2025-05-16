from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from modules.models import MenuItem
from collections import defaultdict

SQLALCHEMY_DATABASE_URL = "sqlite:///./menu.sqlite"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

db = SessionLocal()

def get_db():
    return db

def get_menu_info() :
    items = db.query(MenuItem).all()
    tree = defaultdict(list)
    id_to_name = {}

    for item in items:
        id_to_name[item.id] = item.name
        tree[item.parent_id].append(item.id)

    def render_tree(node_id=None, level=0):
        lines = []
        for child_id in tree[node_id]:
            indent = "  " * level + "- "
            name = id_to_name[child_id]
            lines.append(f"{indent}{name}")
            lines.extend(render_tree(child_id, level + 1))
        return lines

    return "\n".join(render_tree())