const teamLinks = document.querySelectorAll('#team-list li');
const userNameDisplay = document.getElementById('user-name');
const embedContainer = document.getElementById('embed-container');

const replitLinks = {
  chi: "https://replit.com/@chiuser/chi-project?embed=true",
  daryl: "https://replit.com/@daryluser/daryl-project?embed=true",
  imran: "https://replit.com/@imranuser/imran-project?embed=true",
  zach: "https://replit.com/@zachuser/zach-project?embed=true",
  venessa: "https://replit.com/@venessalloyd/venessa-project?embed=true",
  simon: "https://replit.com/@simonuser/simon-project?embed=true",
  joshua: "https://replit.com/@joshuauser/joshua-project?embed=true",
  leon: "https://replit.com/@leonuser/leon-project?embed=true"
};

teamLinks.forEach(link => {
  link.addEventListener('click', () => {
    document.querySelector('li.active').classList.remove('active');
    link.classList.add('active');

    const userKey = link.dataset.user;
    const name = link.textContent;
    userNameDisplay.textContent = name;
    embedContainer.innerHTML = `<iframe src="${replitLinks[userKey]}" frameborder="0" allowfullscreen></iframe>`;
  });
});
