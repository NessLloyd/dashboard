const teamLinks = document.querySelectorAll('#team-list li');
const userNameDisplay = document.getElementById('user-name');
const embedContainer = document.getElementById('embed-container');



teamLinks.forEach(link => {
  link.addEventListener('click', () => {
    document.querySelector('li.active').classList.remove('active');
    link.classList.add('active');

    const userKey = link.dataset.user;
    const name = link.textContent;
    userNameDisplay.textContent = name;
  });
});
