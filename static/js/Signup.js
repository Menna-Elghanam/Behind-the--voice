var Name = document.getElementById("name");
var LastName = document.getElementById("LastName");
var UserEmail = document.getElementById("email");
var UserPassword = document.getElementById("password");
var UserPhone = document.getElementById("phone");
function validateName() {
    var regex = /^[a-zA-Z\s]{2,20}$/;

  if (regex.test(Name.value)) {
    Name.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    Name.classList.add("is-invalid");
    return false;
  }
}
function validateUserPhone() {
  var regex = /^01[0-25][0-9]{8}$/;
  if (regex.test(UserPhone.value)) {
    UserPhone.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    UserPhone.classList.add("is-invalid");
    return false;
  }
}

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
  var regex = /^(?=.*[A-Z])(?=.*\d).{8,20}$/;
  if (regex.test(UserPassword.value)) {
    UserPassword.classList.replace("is-invalid", "is-valid");
    return true;
  } else {
    UserPassword.classList.add("is-invalid");
    return false;
  }
}

function validateUserData() {
  if (
    validateName() &&
    validateUserPhone() &&
    validateUserEmail() &&
    validateUserPassword() == true
  ) {
    return true;
  } else {
    return false;
  }
}

// REDIRECTION
// document.getElementById("SignUpForm").addEventListener("submit", function (event) {
//     event.preventDefault();
//     window.location.href = "login.html";
//   });
