# Getting Help

## Documentation

- [README.md](README.md) — project overview, quick start, API reference
- [docs/architecture.md](docs/architecture.md) — internal architecture, data flow, module map
- [docs/deployment.md](docs/deployment.md) — production deployment, scaling, troubleshooting
- [Swagger Docs](http://localhost:8000/docs) — interactive API documentation (when server is running)

## Community

- [GitHub Discussions](https://github.com/vansh-visariya/ASTRA/discussions) — ask questions, share ideas
- [GitHub Issues](https://github.com/vansh-visariya/ASTRA/issues) — report bugs, request features

## Common Questions

**Q: The dashboard shows "Connection refused"**
A: Make sure the backend server is running (`uvicorn astra.app.server_api:app --port 8000`). Check `dashboard/.env.local` has the correct `NEXT_PUBLIC_API_URL`.

**Q: How do I log in?**
A: Create an account at the sign-up page. Choose `admin` role for full access or `client` role to participate in training. Default admin credentials (if DB initialized): username `admin`, password `admin123` (configurable via `ASTRA_DEFAULT_ADMIN_PASSWORD`).

**Q: Can I use my own model?**
A: Yes. In the dashboard Create Group page, switch to the External tab and enter the Python import path (e.g., `torchvision.models.resnet18`). Or call `POST /api/models/register/architecture` directly.

**Q: My model doesn't seem to train (no accuracy improvement)**
A: Check that the model is registered in the registry, that clients are connected and active, and that enough updates have been received to trigger aggregation (default: 10 updates or 1 second).
