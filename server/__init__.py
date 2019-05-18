from server.server import app, initialize as initialize_server


def run_server(port=3000):
    app.run(port=port)

