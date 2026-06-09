from server.app import app
import uvicorn
from server.config import SERVER_HOST, SERVER_PORT

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
