<template>
  <div>
    <div class="page-heading">
      <div class="row">
        <div class="col-12 col-md-6 order-md-1 order-last">
          <h3>System Updater</h3>
        </div>
        <div class="col-12 col-md-6 order-md-2 order-first">
          <nav aria-label="breadcrumb" class="breadcrumb-header float-start float-lg-end">
            <ol class="breadcrumb">
              <li class="breadcrumb-item">
                  <router-link to="/dashboard">{{ __('dashboard') }}</router-link>
              </li>
              <li class="breadcrumb-item active" aria-current="page">System Updater</li>
            </ol>
          </nav>
        </div>
      </div>

      <div class="row">
        <div class="col-6 col-md-6 order-md-1 order-last">

          <div class="card" v-if="updaterResult && updaterResult.description">
            <div class="card-header">
              <h4>Change Logs of eGrocer {{'v - '+ updaterResult.version }} </h4>
              <span class="pull-right">
                <span class="badge bg-primary" >
                    New Update Available <span class="badge rounded-pill bg-success text-white ms-1">{{'v - '+ updaterResult.version }}</span>
                </span>
              </span>
            </div>
            <div class="card-body">
                <span id="update_description" v-html="updaterResult.description"></span>
            </div>
            <div class="card-footer">
              <b-button @click="systemUpdate" type="button" variant="primary" :disabled="isLoading">
                  <span v-if="isLoading"><b-spinner small label="Spinning"></b-spinner> System is updating... </span>
                  <span v-else>Update The System </span>
              </b-button>
          </div>
          </div>
            <div class="card" v-else>
                <div class="card-header">
                    <h4>New system update is not available. </h4>
                    <span class="pull-right">
                        <span class="badge bg-primary">
                            eGrocer current version is
                            <span class="badge rounded-pill bg-success text-white ms-1">{{ 'v - '+$currentVersion }}</span>
                        </span>
                  </span>
                </div>
            </div>
        </div>
      </div>
    </div>


  </div>
</template>
<script>



import axios from "axios";

export default {
  data: function () {
    return {
        isLoading: false,
        updaterResult: {
            version:'0.0.0'
        }

    };
  },
  computed: {
    sortOptions() {
      // Create an options list from our fields
      return this.fields
        .filter((f) => f.sortable)
        .map((f) => {
          return { text: f.label, value: f.key };
        });
    },
  },

  created: function () {

    this.checkSystemUpdate();

    this.getTimeSlots();
    this.getTimeSlotsSettings();
  },
  methods: {
      checkSystemUpdate(){
          axios.get(this.$baseUrl + '/updater.check').then((response) => {
              let result = JSON.stringify(response.data);
              this.updaterResult = JSON.parse(result);
              console.log("this.updaterResult updater.check => ", this.updaterResult.version);
          });
      },
      systemUpdate(){

          this.$swal.fire({
              title: "Are you Sure?",
              text: "You want be able to revert this",
              confirmButtonText: "Yes, Sure",
              cancelButtonText: "Cancel",
              icon: 'warning',
              showCancelButton: true,
              confirmButtonColor: '#37a279',
              cancelButtonColor: '#d33',
          }).then(result => {
              if (result.value) {

                  this.isLoading = true;
                  axios.get(this.$baseUrl + "/updater.update").then((response) => {
                      if (response != '') {
                          this.showMessage("success", "Congratulation System Updated Successfully");
                          this.checkSystemUpdate();
                          this.isLoading = false;
                          setTimeout(() => {
                              window.location.href = this.$baseUrl + '/migration';
                          }, 5000);
                      } else {
                          this.showMessage("error", "Something went wrong! System did not update.");
                          this.showError("Something went wrong! System did not update.");
                          this.isLoading = false;
                      }
                      console.log("Updater response log =>", response);
                  });
              }
          });

      },
  },
};
</script>
