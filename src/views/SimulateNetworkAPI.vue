<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-12">
        <h3>Build and simulate an electricity distribution network API</h3>
        <p>
          You can find more information on OpenDSS parameters below in the
          <a
            href="http://svn.code.sf.net/p/electricdss/code/trunk/Distrib/Doc/OpenDSSManual.pdf"
            >OpenDSS Manual</a
          >.
        </p>
      </div>
    </div>

    <form id="config" ref="config" @submit.prevent>
      <div class="row">
        <div class="col-sm-12">
          <h5>Electricity distribution network parameters</h5>

          <!-- Experiment parameter: n_lv -->
          <div class="form-group row">
            <label for="n_lv" class="col-md-2 col-form-label">
              Number of LV circuits
            </label>
            <div class="col-md-1">
              <input
                v-model.number="config.n_lv"
                type="number"
                class="form-control"
                id="n_lv"
                placeholder="n_lv"
              />
            </div>
          </div>

          <!-- Experiment parameter: n_id -->
          <div class="form-group row">
            <label for="n_id" class="col-md-2 col-form-label">
              Network ID
            </label>
            <div class="col-md-3">
              <select v-model="config.n_id">
                <option value="1060">11kV urban network</option>
                <option value="1061">11kV urban - rural network</option>
              </select>
            </div>
          </div>

          <!-- DUMMY Experiment parameter: p_ev -->
          <div class="form-group row">
            <label for="p_ev" class="col-md-2 col-form-label">
              Percentage penetration EV
            </label>
            <div class="col-md-1">
              <input
                v-model.number="config.p_ev"
                type="number"
                class="form-control"
                id="p_ev"
                placeholder="p_ev"
              />
            </div>
          </div>

          <!-- DUMMY Experiment parameter: p_pv -->
          <div class="form-group row">
            <label for="p_pv" class="col-md-2 col-form-label">
              Percentage penetration PV
            </label>
            <div class="col-md-1">
              <input
                v-model.number="config.p_pv"
                type="number"
                class="form-control"
                id="p_pv"
                placeholder="p_pv"
              />
            </div>
          </div>

          <!-- DUMMY Experiment upload -->
          <div class="form-group row">
            <label for="c_load" class="col-md-2 col-form-label">
              Custom load profile
            </label>
            <div class="col-md-3">
              <input
                type="file"
                class="form-control"
                id="c_load"
                placeholder="c_load"
              />
            </div>
          </div>
        </div>
      </div>

      <div class="row">
        <div class="col-sm-6">
          <h5>Provided profiles</h5>
        </div>
        <div class="col-sm-6">
          <h5>User custom profiles</h5>
        </div>
      </div>

      <div class="row">
        <div class="col-sm-6">
          <button type="submit" class="btn btn-primary" @click="fetchAPIData">
            Submit
            <template v-if="isLoading">
              <div class="spinner-border spinner-border-sm" role="status"></div>
            </template>
          </button>
        </div>
      </div>

      <div class="row">
        <div class="col-sm-12">
          <h5>Results</h5>
        </div>
      </div>
    </form>

    <template v-if="responseAvailable">
      <div class="mt-3">
        <h4>Plot(s)</h4>
        <div class="card mt-3" style="width: 60rem;">
          <img v-bind:src="'data:image/jpeg;base64,' + plot1" />
        </div>
        <div class="card mt-3" style="width: 60rem;">
          <img v-bind:src="'data:image/jpeg;base64,' + plot2" />
        </div>
        <!--
    <h4>Voltages</h4>
    <div>
      <ol id="voltages">
        <li v-for="(value, index) in voltages" :key="index">
          {{ value }}
        </li>
      </ol>
    </div>
    <h4>Report</h4>
    <div>
      <ul id="report">
        <li v-for="(value, index) in report" :key="index">
          {{ value }}
        </li>
      </ul>
    </div>
    -->
        <div class="mt-3">
          <button class="btn btn-primary" @click="isShowJson = !isShowJson">
            <template v-if="isShowJson">Hide API json response</template>
            <template v-else>Show API json response</template>
          </button>
          <div v-if="isShowJson">
            <code>{{ rawJson }}</code>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script>
export default {
  el: "#main",

  data() {
    return {
      config: {
        n_id: 1060,
        n_lv: 5,
        p_ev: 10,
        p_pv: 10,
        c_loads: ""
      },
      // voltages: [],
      // report: [],
      plot1: "",
      plot2: "",
      rawJson: "empty",
      isShowJson: false,
      isLoading: false,
      responseAvailable: false
    };
  },

  methods: {
    fetchAPIData() {
      // Ask user to wait while API request is formed and made
      this.plot1 = "";
      this.plot2 = "";
      this.rawJson = "wait...";
      this.isLoading = true;
      this.responseAvailable = false;

      var url = process.env.VUE_APP_API_URL + "/simulate";

      if (process.env.NODE_ENV == "development") {
        console.log("Using API URL:", url);
      }

      // If given a non-integer, switch back to 5
      if (this.config.n_lv != Math.round(this.config.n_lv)) {
        this.config.n_lv = 5;
      }
      url += "?n_id=1060&lv_list=1101,1105,1103&dry_run=true";

      let formData = new FormData();
      formData.append("mv_solar_pv_csv", new Blob(["1, 2, 3"], { type: "text/csv" }));
      formData.append("mv_ev_charger_csv", new Blob(["1, 2, 3"], { type: "text/csv" }));
      formData.append("lv_smart_meter_csv", new Blob(["1, 2, 3"], { type: "text/csv" }));
      formData.append("lv_ev_csv", new Blob(["1, 2, 3"], { type: "text/csv" }));
      formData.append("lv_pv_csv", new Blob(["1, 2, 3"], { type: "text/csv" }));
      formData.append("lv_hp_csv", new Blob(["1, 2, 3"], { type: "text/csv" }));

      fetch(url, {
        method: "POST",
        body: formData
      })
        .then(response => {
          if (response.ok) {
            return response.text();
          } else {
            alert(
              "Server returned " + response.status + " : " + response.statusText
            );
          }
        })
        .then(response => {
          // Populate rawJson
          this.rawJson = response;

          // Parse json repsonse
          var responseJson = JSON.parse(response);

          // // Parse revoltages port from json to array to list
          // this.voltages = responseJson["voltages"];

          // // Parse report from json to array to list
          // this.report = responseJson["report"];

          // Parse plot from json to image data
          this.plot1 = responseJson["plot1"];
          this.plot2 = responseJson["plot2"];

          this.responseAvailable = true;
          this.isLoading = false;
        })
        .catch(err => {
          console.log(err);
        });
    }
  }
};
</script>

<style scoped>
.form-check {
  margin-bottom: 15px;
}
#displayBox {
  margin-top: 20px;
}
</style>
