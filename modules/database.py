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

def get_menu_info_for_prompt():
    items = db.query(MenuItem).all()
    
    # 메뉴 아이템들을 딕셔너리로 변환
    menu_dict = {}
    tree = defaultdict(list)
    
    for item in items:
        menu_dict[item.id] = {
            'id': item.id,
            'parent_id': item.parent_id,
            'name': item.name,
            'description': item.description,
            'emoji': item.emoji,
            'keywords': item.keywords if item.keywords else []
        }
        tree[item.parent_id].append(item.id)
    
    def build_hierarchy_text(node_id=None, level=0):
        """계층 구조를 텍스트로 렌더링"""
        lines = []
        for child_id in tree[node_id]:
            item = menu_dict[child_id]
            indent = "  " * level + "- "
            
            # 기본 정보
            line = f"{indent}{item['name']}"
            
            # 설명이 있으면 추가
            if item['description']:
                line += f" ({item['description']})"
            
            # 키워드가 있으면 추가
            if item['keywords']:
                keywords_str = ", ".join(item['keywords'])
                line += f" [키워드: {keywords_str}]"
            
            lines.append(line)
            lines.extend(build_hierarchy_text(child_id, level + 1))
        return lines
    
    # 전체 메뉴 구조를 텍스트로 변환
    hierarchy_text = "\n".join(build_hierarchy_text())
    
    return {
        'hierarchy_text': hierarchy_text,
        'menu_items': menu_dict,
        'tree_structure': dict(tree)
    }
