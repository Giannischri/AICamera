const navbarList = document.querySelector('.flashing-navbar ul');
const leds = navbarList.querySelectorAll('li');
var button = document.getElementById("mybtn");
const video=new EventSource('/')
// Function to toggle flashing for a list item
function toggleFlashing(listItem) {
  const isFlashing = listItem.dataset.flashing === 'true';
  listItem.dataset.flashing = isFlashing ? 'false' : 'true';
}

// Add event listeners to toggle flashing on click (or any other condition)
listItems.forEach(listItem => {
  listItem.addEventListener('click', () => toggleFlashing(listItem));
});
const eventSource = new EventSource('/stream_data');

// Event listener for receiving messages from the server
eventSource.onmessage = function(event) {
    const eventData = event.data;

    // Check if the message contains metadata or frame data
    print(event.data)
};

// Event listener for handling errors
eventSource.onerror = function(event) {
    console.error('EventSource error:', event);
};
function handleClick(param) {

fetch('/faulty-detect', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'cameraId':param
    },
})
.then(response => {
    if (response.ok) {
        return response.json();
    }
    throw new Error('Network response was not ok.');
})
.then(data => {
    console.log('Response from Flask:', data);
})
.catch(error => {
    console.error('There was a problem with the fetch operation:', error);
});
}
// Example: Simulate flashing for number 5 after 3 seconds
setTimeout(() => toggleFlashing(listItems[4]), 3000); // Target the 5th list item (index 4)
function getleds() {
    fetch('/leds')
        .then(response => response.json())
        .then(data => {
            // Process the data received from the backend
            print(data)
            // var btn = document.getElementById('btn');
            // var loopIndexValue = btn.getAttribute('data-loop-index');
            // loopIndexValue = parseInt(loopIndexValue);
            // toggleFlashing(listItems.item(loopIndexValue));
            // print(loopIndexValue);

        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}