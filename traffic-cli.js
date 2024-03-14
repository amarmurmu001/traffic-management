const readline = require('readline');

// Create interface for input/output
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Function to ask for vehicle numbers
function askForVehicles(lane) {
  return new Promise(resolve => {
    rl.question(`Enter number of two-wheelers and four-wheelers for Lane ${lane}, separated by a space: `, (input) => {
      const [twoWheelers, fourWheelers] = input.split(' ').map(Number);
      resolve({ twoWheelers, fourWheelers, totalVehicles: twoWheelers + (fourWheelers * 2) });
    });
  });
}

// Main function to process the lanes
async function processLanes() {
  const lanes = {};

  // Get vehicle numbers for each lane
  for (let i = 1; i <= 4; i++) {
    lanes[`lane${i}`] = await askForVehicles(i);
  }

  // Sort lanes by density
  const sortedLanes = Object.entries(lanes)
                            .sort(([, a], [, b]) => b.totalVehicles - a.totalVehicles)
                            .map(([lane, ]) => lane);

  // Display sorted lanes
  console.log(`Lanes in order of descending vehicle density: ${sortedLanes.join(' > ')}`);
  
  rl.close();
}

// Start the process
processLanes();
