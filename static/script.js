const teamLinks = document.querySelectorAll('#team-list li');
const userNameDisplay = document.getElementById('user-name');
const embedContainer = document.getElementById('embed-container');

teamLinks.forEach(link => {
  link.addEventListener('click', () => {
    // Update active state
    document.querySelector('li.active').classList.remove('active');
    link.classList.add('active');

    const userKey = link.dataset.user;
    const name = link.textContent;
    userNameDisplay.textContent = name;

    // Dynamic iframe loaders per person
    switch (userKey) {
      case "imran":
        embedContainer.innerHTML = `
          <iframe src="/templates" frameborder="0" width="100%" height="700px"></iframe>
        `;
        break;

      case "chi":
        embedContainer.innerHTML = `<p>Chi’s project is not embedded yet. Coming soon!</p>`;
        break;

      case "daryl":
        embedContainer.innerHTML = `<p>Daryl’s project is not embedded yet. Coming soon!</p>`;
        break;

      case "zach":
        embedContainer.innerHTML = `<p>Zach’s project is not embedded yet. Coming soon!</p>`;
        break;

      case "vanessa":
        embedContainer.innerHTML = `<p>Vanessa’s project is not embedded yet. Coming soon!</p>`;
        break;

      case "simon":
        embedContainer.innerHTML = `<p>Simon’s project is not embedded yet. Coming soon!</p>`;
        break;

      case "joshua":
        embedContainer.innerHTML = `<p>Joshua and Vireak’s project is not embedded yet. Coming soon!</p>`;
        break;

      case "leon":
        embedContainer.innerHTML = `<p>Leon’s project is not embedded yet. Coming soon!</p>`;
        break;

      default:
        embedContainer.innerHTML = `<p>No content available for this user.</p>`;
    }
  });
});
