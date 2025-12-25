# Using Google Colab in VS Code

You can run these notebooks directly inside VS Code while using Google Colab's free GPUs as the backend.

## Setup

1. **Install Extension**:
   - Install the [Google Colab for VS Code](https://marketplace.visualstudio.com/items?itemName=google.colab) extension (by Google).

2. **Connect**:
   - Open `notebooks/00_onboarding.ipynb`.
   - In the top-right kernel picker, select **Connect to Google Colab**.
   - Authenticate with your Google account if prompted.
   - Choose a runtime (ensure it has GPU enabled if possible, or use the "Change Runtime Type" command in the palette).
   - *Note*: You might need to set the runtime type to GPU via the command palette: `> Google Colab: Change Runtime Type`.

3. **Running**:
   - Run cells normally.
   - The notebook has been updated to automatically clone the repo source code into the Colab VM if it detects it's missing, ensuring imports verify correctly.

## Troubleshooting

- **"Module src not found"**: Ensure you ran the cell that checks for repo existence. In VS Code connected to Colab, your local file system is NOT mounted to the VM. The notebook handles this by cloning the repo into the VM instance.
- **Connection dropped**: Colab timeouts apply. You may need to reconnect.
