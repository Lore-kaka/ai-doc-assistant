"""数据库初始化脚本

使用方法:
    python scripts/init_db.py
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 添加项目根目录到 Python 路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from app.rag.retriever import init_database
    
    print("=" * 50)
    print("🔧 初始化向量数据库")
    print("=" * 50)
    
    try:
        db = init_database()
        
        # 获取数据库大小
        count = db._collection.count()
        
        print("\n" + "=" * 50)
        print("✅ 数据库初始化成功")
        print(f"📊 记录数: {count} 条")
        print("=" * 50)
    
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
