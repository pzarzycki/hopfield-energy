# Energy-Based Memory Models

A comprehensive interactive showcase of energy-based memory models and their evolution, from classical Hopfield Networks to modern variants.

🚀 **Live Demo:** [https://pzarzycki.github.io/hopfield-energy/](https://pzarzycki.github.io/hopfield-energy/)

## Overview

This project provides an educational exploration of energy-based models (EBMs) in machine learning, covering:

- **Introduction** - Overview of energy-based models and core concepts
- **Hopfield Networks** - Classical associative memory systems
- **Boltzmann Machines** - Stochastic neural networks with hidden units
- **Restricted Boltzmann Machines** - Practical variant with bipartite structure
- **Dense Hopfield Networks** - Modern networks with exponential capacity
- **Multi-layer RBMs** - Deep Belief Networks and layer-wise pre-training
- **Energy-Based Models** - General framework and modern applications

## Features

✨ **Modern Design** - Clean, professional UI with smooth animations  
📱 **Fully Responsive** - Works seamlessly on desktop, tablet, and mobile  
🎨 **Sticky Navigation** - Left sidebar navigation for easy browsing  
⚡ **Fast Performance** - Built with Vite for optimal load times  
📚 **Comprehensive Content** - Detailed explanations with mathematical formulations

## Technology Stack

- **React 19** with TypeScript
- **React Router** for client-side routing
- **Vite** for blazing-fast development and building
- **CSS3** with custom properties for theming
- **GitHub Actions** for automated deployment

## Local Development

### Prerequisites

- Node.js 20 or later
- npm or yarn

### Setup

```bash
# Clone the repository
git clone https://github.com/pzarzycki/hopfield-energy.git
cd hopfield-energy

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173/`

### Build for Production

```bash
# Create production build
npm run build

# Preview production build
npm run preview
```

## Deployment

The project is automatically deployed to GitHub Pages when changes are pushed to the `main` branch. The deployment workflow:

1. Builds the React application
2. Uploads the `dist` folder as a Pages artifact
3. Deploys to GitHub Pages

### Manual Deployment

```bash
# Build the project
npm run build

# The dist folder is ready for deployment
```

## Project Structure

```
hopfield-energy/
├── src/
│   ├── components/       # React components
│   │   ├── Layout.tsx    # Main layout wrapper
│   │   └── Sidebar.tsx   # Navigation sidebar
│   ├── pages/            # Page components
│   │   ├── Introduction.tsx
│   │   ├── HopfieldNetworks.tsx
│   │   ├── BoltzmannMachines.tsx
│   │   ├── RestrictedBoltzmannMachines.tsx
│   │   ├── DenseHopfieldNetworks.tsx
│   │   ├── MultiLayerRBMs.tsx
│   │   └── EnergyBasedModels.tsx
│   ├── App.tsx           # Main app component with routing
│   ├── App.css           # Layout and component styles
│   ├── index.css         # Global styles and design system
│   └── main.tsx          # Application entry point
├── .github/
│   └── workflows/
│       └── deploy.yml    # GitHub Actions deployment
├── vite.config.ts        # Vite configuration
└── package.json
```

## Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features or content
- Submit pull requests

## License

MIT License - feel free to use this project for educational purposes.

## Acknowledgments

This project was created to provide an accessible educational resource on energy-based models, drawing from:

- Classical papers by Hopfield, Hinton, and others
- Modern research on Deep Hopfield Networks
- The broader machine learning community's contributions

---

Built with ❤️ using React and Vite
