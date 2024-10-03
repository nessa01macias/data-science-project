document.addEventListener('DOMContentLoaded', function () {
    console.log("DOM fully loaded and parsed");

    const form = document.querySelector('form');

    form.addEventListener('submit', function (event) {
        console.log("Form submission event triggered");

        const reviewTextarea = document.querySelector('textarea[name="Review"]');
        const reviewText = reviewTextarea.value.trim();

        if (reviewText === '') {
            alert('Please enter a review before submitting.');
            event.preventDefault(); // Prevent form submission
        }
    });
});

