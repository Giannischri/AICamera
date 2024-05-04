const ledcontainer = document.getElementById('ledcontainer')
const container=document.querySelector('.container');
var founditem;// Function to toggle flashing for a list item
var ITEMS=[];

// Add event listeners to toggle flashing on click (or any other condition)
//listItems.forEach(listItem => {
  //listItem.addEventListener('click', () => toggleFlashing(listItem));
//});
var socket = io.connect();
socket.on('foundfall', function(data) {
   num=data.name.substring(4);
   setTimeout(function() {
    var alarmSound = document.getElementById('alarmSound');
    alarmSound.play();
    }, 1000);

   toggleFlashing(num);
    ITEMS[num]=data;
    console.log(ITEMS)
    flashingcord(num)
    console.log("sleep is over")

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
    const isFlashing=led.classList.contains('flashing');
    console.log(isFlashing);
    if (isFlashing === true) {
      led.classList.remove('flashing');
    } else {
       led.classList.add('flashing');

    }
}
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
async function flashingcord(num) {
    await sleep(5000);
    toggleFlashing(num);

}

