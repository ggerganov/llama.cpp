'''
    Memory tools that use sqlite-vec as a vector database (combined w/ sqlite-lembed or sqlite-rembed for embeddings).

    Note: it's best to run this in a silo w/:
        
        ./examples/agent/serve_tools_inside_docker.sh

    # Run w/o other tools:
    
    ## Prerequisites:
    
    pip install aiosqlite "fastapi[standard]" sqlite-lembed sqlite-rembed sqlite-vec uvicorn
    
    ## Usage w/ sqlite-rembed:
    
    ./llama-server --port 8081 -fa -c 0 --embeddings --rope-freq-scale 0.75 \
        -hfr nomic-ai/nomic-embed-text-v1.5-GGUF -hff nomic-embed-text-v1.5.Q4_K_M.gguf
    MEMORY_SQLITE_DB=memory_rembed.db \
        EMBEDDINGS_DIMS=768 \
        EMBEDDINGS_ENDPOINT=http://localhost:8081/v1/embeddings \
        python examples/agent/tools/memory.py
        
    ## Usage w/ sqlite-lembed:
    
    MEMORY_SQLITE_DB=memory_lembed.db \
        EMBEDDINGS_DIMS=768 \
        EMBEDDINGS_MODEL_FILE=~/Library/Caches/llama.cpp/nomic-embed-text-v1.5.Q4_K_M.gguf \
        python examples/agent/tools/memory.py

    ## Test:
    
    curl -X POST "http://localhost:8000/memorize" -H "Content-Type: application/json" -d '["User is Olivier Chafik", "User is a Software Engineer"]'
    curl -X POST "http://localhost:8000/search_memory?text=What%20do%20we%20do%3F"
'''

import logging
import aiosqlite
import fastapi
import os
import sqlite_lembed
import sqlite_rembed
import sqlite_vec

verbose = os.environ.get('VERBOSE', '0') == '1'
db_path = os.environ['MEMORY_SQLITE_DB']


# Embeddings configuration:
# Can either provide an embeddings model file (to be loaded locally by sqlite-lembed)
# or an embeddings endpoint w/ optional api key (to be queried remotely by sqlite-rembed).
embeddings_dims = int(os.environ['EMBEDDINGS_DIMS'])
if 'EMBEDDINGS_MODEL_FILE' in os.environ:
    local = True
    embed_fn = 'lembed'
    embeddings_model_file = os.environ['EMBEDDINGS_MODEL_FILE']
    logging.info(f'Using local embeddings model: {embeddings_model_file}')
elif 'EMBEDDINGS_ENDPOINT' in os.environ:
    local = False
    embed_fn = 'rembed'
    embeddings_endpoint = os.environ['EMBEDDINGS_ENDPOINT']
    embeddings_api_key = os.environ.get('EMBEDDINGS_API_KEY')
    logging.info(f'Using remote embeddings endpoint: {embeddings_endpoint}')
else:
    raise ValueError('Either EMBEDDINGS_MODEL_FILE or EMBEDDINGS_ENDPOINT must be set')


async def setup_db(db: aiosqlite.Connection):
    
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    if local:
        await db.load_extension(sqlite_lembed.loadable_path())
    else:
        await db.load_extension(sqlite_rembed.loadable_path())
    await db.enable_load_extension(False)

    client_name = 'default'
    
    if local:
        await db.execute(f'''
            INSERT INTO lembed_models(name, model) VALUES (
                '{client_name}', lembed_model_from_file(?)
            );
        ''', (embeddings_model_file,))
    else:
        await db.execute(f'''
            INSERT INTO rembed_clients(name, options) VALUES (
                '{client_name}', rembed_client_options('format', 'llamafile', 'url', ?, 'key', ?)
            );
        ''', (embeddings_endpoint, embeddings_api_key))
        
    async def create_vector_index(table_name, text_column, embedding_column):
        '''
            Create an sqlite-vec virtual table w/ an embedding column
            kept in sync with a source table's text column.
        '''

        await db.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_{embedding_column} USING vec0(
                {embedding_column} float[{embeddings_dims}]
            )
        ''')
        await db.execute(f'''
            CREATE TRIGGER IF NOT EXISTS insert_{table_name}_{embedding_column}
            AFTER INSERT ON {table_name}
            BEGIN
                INSERT INTO {table_name}_{embedding_column} (rowid, {embedding_column})
                VALUES (NEW.rowid, {embed_fn}('{client_name}', NEW.{text_column}));
            END;
        ''')
        await db.execute(f'''
            CREATE TRIGGER IF NOT EXISTS update_{table_name}_{embedding_column}
            AFTER UPDATE OF {text_column} ON {table_name}
            BEGIN
                UPDATE {table_name}_{embedding_column}
                SET {embedding_column} = {embed_fn}('{client_name}', NEW.{text_column})
                WHERE rowid = NEW.rowid;
            END;
        ''')
        await db.execute(f'''
            CREATE TRIGGER IF NOT EXISTS delete_{table_name}_{embedding_column}
            AFTER DELETE ON {table_name}
            BEGIN
                DELETE FROM {table_name}_{embedding_column}
                WHERE rowid = OLD.rowid;
            END;
        ''')
        def search(text: str, top_n: int, columns: list[str] = ['rowid', text_column]):
            '''
                Search the vector index for the embedding of the provided text and return
                the distance of the top_n nearest matches + their corresponding original table's columns.
            '''

            col_seq = ', '.join(['distance', *(f"{table_name}.{c}" for c in columns)])
            return db.execute(
                f'''
                    SELECT {col_seq}
                    FROM (
                        SELECT rowid, distance
                        FROM {table_name}_{embedding_column}
                        WHERE {table_name}_{embedding_column}.{embedding_column} MATCH {embed_fn}('{client_name}', ?)
                        ORDER BY distance
                        LIMIT ?
                    )
                    JOIN {table_name} USING (rowid)
                ''',
                (text, top_n)
            ) 
        return search

    await db.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL
        )
    ''')
    facts_search = await create_vector_index('facts', 'content', 'embedding')
    
    await db.commit()
    
    return dict(
        facts_search=facts_search,
    )


async def memorize(facts: list[str]):
    'Memorize a set of statements / facts.'

    async with aiosqlite.connect(db_path) as db:
        await setup_db(db)
        await db.executemany(
            'INSERT INTO facts (content) VALUES (?)',
            [(fact,) for fact in facts]
        )
        await db.commit()


async def search_memory(text: str, top_n: int = 10):
    'Search the memory for the closest informations to the provided text (return only the top_n best matches).'

    async with aiosqlite.connect(db_path) as db:
        db_functions = await setup_db(db)
        async with db_functions['facts_search'](text, top_n) as cursor:
            # Return a json array of objects w/ columns
            results = await cursor.fetchall()
            cols = [c[0] for c in cursor.description]
            return [dict(zip(cols, row)) for row in results]
   

# This main entry point is just here for easy debugging
if __name__ == '__main__':
    import uvicorn

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    app = fastapi.FastAPI()
    app.post('/memorize')(memorize)
    app.post('/search_memory')(search_memory)
    uvicorn.run(app)
