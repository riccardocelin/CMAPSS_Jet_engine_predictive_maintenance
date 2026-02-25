import uvicorn

def start_local_server():

    print('Starting uvicorn server ...')       

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        reload=True,
    )
    # webbrowser.open("http://127.0.0.1:8000")


def main():
    start_local_server()

if __name__ == '__main__':
    main()