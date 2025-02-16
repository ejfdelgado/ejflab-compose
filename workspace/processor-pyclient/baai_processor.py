import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor
import asyncpg


class BaaiProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {}

    async def pgtest(self, args, default_arguments):
        conn_params = {
            'user': os.environ['POSTGRES_USER'],
            'password': os.environ['POSTGRES_PASSWORD'],
            'database': os.environ['POSTGRES_DB'],
            'host': os.environ['POSTGRES_HOST'], 
            'port': int(os.environ['POSTGRES_PORT'])
        }

        # Establish a connection
        conn = await asyncpg.connect(**conn_params)

        #await conn.execute('''INSERT INTO users(name, dob) VALUES($1)''', 
        #                   'Bob')

        row = await conn.fetchrow(
        "SELECT NOW(), $1 as text", 'Bob')

        print(row)

        await conn.close()
        return {}

    async def process(self, args, default_arguments):
        method = args['method']
        if method == "pgtest":
            return await self.pgtest(args, default_arguments)


async def main():
    processor = BaaiProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
