# 🌐 ParkVision Railway Deployment Guide

Deploy your ParkVision parking detection system for **completely FREE** on Railway.

## 🚀 Railway Deployment (Recommended) 🚂

**Free Tier:** 500 hours/month + $5 credit monthly

### Why Railway?
- ✅ **Easiest deployment** - Just connect GitHub and deploy
- ✅ **Automatic HTTPS** - Secure URLs out of the box
- ✅ **Auto-deploy** - Updates when you push to GitHub
- ✅ **Custom domains** - Use your own domain name
- ✅ **No CLI required** - Everything through web interface
- ✅ **Great free tier** - More than enough for 24/7 operation

### 📋 Steps to Deploy:

#### 1. Push Code to GitHub
First, make sure your code is on GitHub:
```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

#### 2. Deploy on Railway
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** account
3. **New Project** → **Deploy from GitHub repo**
4. **Select** your ParkVision repository
5. **Railway auto-detects** your Dockerfile
6. **Deploy automatically** - takes 2-3 minutes
7. **Get your live URL** like: `https://parkvision-production.up.railway.app`

#### 3. Configuration
Railway automatically handles:
- Port configuration (reads from `railway.json`)
- Environment variables
- Health checks
- HTTPS certificates
- Auto-scaling

### 🌍 Access Your Live App

Once deployed, your ParkVision API will be available at your Railway URL:

**Main Endpoints:**
- `https://your-app.up.railway.app/` - API information
- `https://your-app.up.railway.app/snapshot` - Current parking status
- `https://your-app.up.railway.app/health` - Health check
- `https://your-app.up.railway.app/stream` - Real-time updates

**Example Response:**
```json
{
  "total_spots": 50,
  "occupied_spots": 23,
  "free_spots": 27,
  "detection_confidence": 0.85,
  "timestamp": "2025-08-05T10:45:00.000Z",
  "path_data": {
    "has_path": true,
    "path_length": 4,
    "target_spot": 12
  }
}
```

### 💡 Railway Tips

1. **Memory:** Your app gets 512MB-1GB RAM (perfect for YOLO)
2. **Cold Start:** First request takes 30-60 seconds (model loading)
3. **Sleep Mode:** App sleeps after 5 minutes of inactivity (free tier)
4. **Keep Alive:** Use `/health` endpoint to keep app running
5. **Logs:** View real-time logs in Railway dashboard

### 🔧 Configuration Files

The following files configure Railway deployment:

- `railway.json` - Railway platform configuration
- `Dockerfile` - Container build instructions
- `docker-compose.yml` - Local development setup
- `.dockerignore` - Files to exclude from Docker build

### 🔒 Security & Performance

- ✅ CORS enabled for cross-origin requests
- ✅ Health checks configured (`/health` endpoint)
- ✅ Proper error handling
- ✅ Background processing for real-time detection
- ✅ Automatic scaling support
- ✅ HTTPS enforced by default

### 🆓 Free Tier Details

**Railway Free Plan includes:**
- **500 execution hours/month** (enough for 24/7 if optimized)
- **$5 credit monthly** (for additional usage)
- **1GB memory** per service
- **1 vCPU** per service
- **Unlimited deployments**
- **Custom domains**
- **Automatic HTTPS**

### 🎯 After Deployment

1. **Test your endpoints** using the provided URLs
2. **Monitor usage** in Railway dashboard
3. **Set up monitoring** using the `/health` endpoint
4. **Configure custom domain** (optional)
5. **Set environment variables** if needed

### 🚀 Going Live

Your **ParkVision parking detection system** will be:
- ✅ **Live 24/7** (with proper configuration)
- ✅ **Globally accessible** via HTTPS
- ✅ **Auto-updating** when you push code changes
- ✅ **Monitoring parking** in real-time
- ✅ **Providing APIs** for mobile apps and dashboards

**Deploy in under 5 minutes and have your parking system live worldwide!** 🌐✨
