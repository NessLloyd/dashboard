// script.js

document.addEventListener('DOMContentLoaded', () => {
    const openBtns = document.querySelectorAll('.open-btn');

    openBtns.forEach((btn) => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = btn.getAttribute('data-section-id');
            const filesContainer = document.getElementById(sectionId + '-files');

            // Load and display attached files
            // Replace this with your actual file loading logic
            const files = [
                { name: 'File 1', url: 'Chi.pdf' },
                { name: 'File 2', url: 'Daryl.pdf' },
                { name: 'File 3', url: 'Imran.pdf' },
                { name: 'File 4', url: 'Zach.pdf' },
                { name: 'File 5', url: 'Vanessa.pdf' },
                { name: 'File 6', url: 'Simon.pdf' },
                { name: 'File 7', url: 'Joshua_Vireak.pdf' },
                { name: 'File 8', url: 'Leon.pdf' }
            ];

            files.forEach((file) => {
                const fileLink = document.createElement('a');
                fileLink.href = file.url;
                fileLink.download = file.name;
                fileLink.click();

                // Create a link to view the file content
                const viewLink = document.createElement('a');
                viewLink.href = file.url;
                viewLink.target = '_blank';
                viewLink.textContent = 'View File';

                filesContainer.appendChild(viewLink);
            });
        });
    });
});
