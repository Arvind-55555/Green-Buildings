# GitHub Pages Deployment Guide

## Deployment Status

Your web page has been deployed to GitHub Pages!

## Access Your Live Dashboard

**Live URL:** https://arvind-55555.github.io/Green-Buildings/

## What Was Deployed

1. **Main Dashboard**: `index.html` - Interactive visualization dashboard
2. **Visualization Images**: All PNG files from `results/` directory
3. **Dataset**: UCI Energy Efficiency dataset
4. **Documentation**: README and other documentation files

## Setup Instructions

### Automatic Deployment (GitHub Actions)

A GitHub Actions workflow has been configured to automatically deploy your site when you push to the `main` branch.

### Manual Setup (If Needed)

1. Go to your repository: https://github.com/Arvind-55555/Green-Buildings
2. Click on **Settings** tab
3. Scroll down to **Pages** section (left sidebar)
4. Under **Source**, select:
   - **Branch**: `main`
   - **Folder**: `/ (root)`
5. Click **Save**

Your site will be available at: `https://arvind-55555.github.io/Green-Buildings/`

## File Structure for GitHub Pages

```
Green-Buildings/
├── index.html              # Main dashboard (GitHub Pages entry point)
├── visualizations.html     # Alternative access
├── results/                # Visualization images
│   ├── *.png              # All visualization charts
├── data/                   # Dataset files
├── README.md               # Project documentation
└── .nojekyll              # Disable Jekyll processing
```

## Updating the Site

To update your GitHub Pages site:

```bash
# Make changes to files
# Then commit and push
git add .
git commit -m "Update visualizations"
git push origin main
```

GitHub Pages will automatically rebuild and deploy your site (usually takes 1-2 minutes).

## Troubleshooting

### Site Not Loading?

1. **Check GitHub Pages Status**:
   - Go to repository Settings → Pages
   - Verify source branch is set to `main`
   - Check if there are any build errors

2. **Verify Files Are Pushed**:
   ```bash
   git ls-remote origin main
   ```

3. **Check File Paths**:
   - Ensure `index.html` exists in root
   - Verify `results/` folder contains PNG files
   - Check image paths in HTML are relative (e.g., `results/image.png`)

4. **Wait for Deployment**:
   - First deployment can take 5-10 minutes
   - Subsequent updates take 1-2 minutes

### Images Not Showing?

- Ensure PNG files are committed to repository
- Check that `.gitignore` doesn't exclude `results/*.png`
- Verify image paths in HTML match actual file locations

### 404 Error?

- Make sure `index.html` is in the root directory
- Verify branch name is `main` (not `master`)
- Check GitHub Pages settings point to correct branch

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file in root with your domain:
   ```
   yourdomain.com
   ```

2. Configure DNS settings as per GitHub Pages documentation

## Repository Links

- **Repository**: https://github.com/Arvind-55555/Green-Buildings
- **Live Site**: https://arvind-55555.github.io/Green-Buildings/
- **Issues**: https://github.com/Arvind-55555/Green-Buildings/issues

## Next Steps

1. ✅ Code pushed to GitHub
2. ✅ GitHub Actions workflow configured
3. ⏳ Enable GitHub Pages in repository settings (if not automatic)
4. ⏳ Wait for initial deployment (5-10 minutes)
5. ✅ Access your live dashboard!

---

**Note**: If GitHub Pages doesn't activate automatically, you may need to enable it manually in the repository Settings → Pages section.

