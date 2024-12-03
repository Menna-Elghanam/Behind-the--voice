// script.js

document.addEventListener("DOMContentLoaded", () => {
    const circleContainer = document.getElementById('circle-container');

    // Callback function for the Intersection Observer
    const observerCallback = (entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                circleContainer.classList.add('animate');
                
                // observer.disconnect(); // Stop observing after the animation starts
            }
            else {
                circleContainer.classList.remove('animate');
              }
        });
    };

    // Options for the observer
    const observerOptions = {
        threshold: 0.5  // Trigger when 50% of the element is visible
    };

    // Create the Intersection Observer
    const observer = new IntersectionObserver(observerCallback, observerOptions);

    // Start observing the circle container
    observer.observe(circleContainer);
});
