# Use official Node.js image
FROM node:18.18.2

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package.json .
COPY package-lock.json .
RUN npm install

# Copy application code
COPY . .

# Expose port and start the development server
EXPOSE 3000
CMD ["npm", "start"]