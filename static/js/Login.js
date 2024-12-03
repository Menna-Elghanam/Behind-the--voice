var UserEmail = document.getElementById("UserEmail");
var UserPassword = document.getElementById("UserPassword");
function validateUserEmail() {
  var regex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

  if (regex.test(UserEmail.value)) {
    UserEmail.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    UserEmail.classList.add("is-invalid");
    return false;
  }
}
function validateUserPassword() {
  var regex = /^(?=.*[A-Z])(?=.*\d).{8,25}$/;
  if (regex.test(UserPassword.value)) {
    UserPassword.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    UserPassword.classList.add("is-invalid");
    return false;
  }
}

function validateUserData() {
  if (validateUserEmail() && validateUserPassword() == true) {
    return true;
  } else {
    return false;
  }
}
