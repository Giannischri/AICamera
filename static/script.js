const ledcontainer = document.getElementById('ledcontainer')
const container=document.querySelector('.container');
var founditem;// Function to toggle flashing for a list item


// Add event listeners to toggle flashing on click (or any other condition)
//listItems.forEach(listItem => {
  //listItem.addEventListener('click', () => toggleFlashing(listItem));
//});
var socket = io();
socket.on('found', function(data) {
    numOfItem=parseInt(data.substring(4));
    toggleFlashing(numOfItem)
    togglevisibility(numOfItem)
});

function handleClick(param) {
    if(param===founditem[17])
        socket.emit('save',founditem);
}
// Example: Simulate flashing for number 5 after 3 seconds
//setTimeout(() => toggleFlashing(listItems[4]), 3000); // Target the 5th list item (index 4)
function togglevisibility(data){
    griditem=container.querySelector('.grid-item:nth-child('+numOfItem+')');
    if(griditem.querySelector('.button2').style.visibility==='hidden')
        griditem.querySelector('.button2').style.visibility = 'visible';
    else
        griditem.querySelector('.button2').style.visibility = 'hidden';
}
function toggleFlashing(listItem) {
    console.log(listItem)
    var led = ledcontainer.querySelector('li:nth-child(' + listItem + ')');
    const isFlashing=led.getAttribute('data-flashing');
    if (isFlashing === 'true') {
      led.classList.remove('flashing');
    } else {
       led.classList.add('flashing');

    }
}

