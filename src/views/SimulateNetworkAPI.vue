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
        <div class="col-lg-12">
          <h4>Electricity distribution network parameters</h4>
        </div>

        <div class="col-lg-6">
          <h5>Medium voltage (MV)</h5>

          <div class="form-group row">
            <!-- Experiment parameter: n_id, network ID -->
            <label for="n_id" class="col-sm-6 col-form-label">
              Network ID
            </label>
            <div class="col-sm-6">
              <select v-model="config.n_id" class="form-control">
                <option value="1060">11kV urban network</option>
                <option value="1061">11kV urban - rural network</option>
              </select>
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: xfmr_scale, MV transformer scaling -->
            <label for="xfmr_scale" class="col-md-6 col-form-label">
              MV transformer scaling
            </label>
            <div class="col-md-6">
              <input
                v-model.number="config.xfmr_scale"
                type="float"
                class="form-control"
                id="xfmr_scale"
                placeholder="xfmr_scale"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: oltc_setpoint -->
            <label for="oltc_setpoint" class="col-md-6 col-form-label">
              MV transformer on-load tap charger (OLTC) set point
            </label>
            <div class="col-md-6">
              <input
                v-model.number="config.oltc_setpoint"
                type="float"
                class="form-control"
                id="oltc_setpoint"
                placeholder="OLTC set point, e.g. 1.04"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: oltc_bandwidth -->
            <label for="oltc_bandwidth" class="col-md-6 col-form-label">
              MV transformer on-load tap charger (OLTC) bandwidth
            </label>
            <div class="col-md-6">
              <input
                v-model.number="config.oltc_bandwidth"
                type="float"
                class="form-control"
                id="oltc_bandwidth"
                placeholder="OLTC bandwidth, e.g. 0.13"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: rs_pen -->
            <label for="rs_pen" class="col-md-6 col-form-label">
              Percentage residential loads
            </label>
            <div class="col-md-6">
              <input
                v-model.number="config.rs_pen"
                type="float"
                class="form-control"
                id="rs_pen"
                placeholder="Percentage residential loads e.g. 0.8"
              />
            </div>
          </div>
        </div>

        <div class="col-lg-6">
          <h5>Low voltage (LV)</h5>

          <div class="form-group row">
            <!-- Experiment parameter: lv_default (if custom, open lv_list option below) -->
            <label for="lv_default" class="col-md-6 col-form-label">
              IDs of up to 5 LV networks to model in detail
            </label>
            <div class="col-md-6">
              <select v-model="config.lv_default" class="form-control">
                <option>1101, 1105, 1103</option>
                <option>1101, 1102, 1103</option>
                <option>Custom</option>
              </select>
            </div>
          </div>

          <template v-if="config.lv_default == 'Custom'">
            <div class="form-group row">
              <!-- Experiment parameter: lv_list -->
              <label for="lv_list" class="col-md-6 col-form-label">
                Custom selection of up to 5 LV network IDs to model in detail
              </label>
              <div class="col-md-6">
                <input
                  v-model.number="config.lv_list"
                  type="string"
                  class="form-control"
                  id="lv_list"
                  placeholder="1101, 1105, 1103"
                />
              </div>
            </div>
          </template>
        </div>
      </div>

      <div class="row">
        <div class="col-lg-6">
          <h5>Provided profiles</h5>

          <!-- DUMMY Experiment upload -->
          <div class="form-group row">
            <label for="c_load" class="col-md-6 col-form-label">
              Custom load profile
            </label>
            <div class="col-md-6">
              <input
                type="file"
                class="form-control"
                id="c_load"
                placeholder="c_load"
              />
            </div>
          </div>
        </div>

        <div class="col-lg-6">
          <h5>User custom profiles</h5>
        </div>
      </div>

      <div class="row">
        <div class="col-lg-12">
          <button type="submit" class="btn btn-primary" @click="fetchAPIData">
            Submit
            <template v-if="isLoading">
              <div class="spinner-border spinner-border-sm" role="status"></div>
            </template>
          </button>
        </div>
      </div>

      <div class="row">
        <div class="col-lg-12">
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
        xfmr_scale: 1.0,
        oltc_setpoint: 1.04,
        oltc_bandwidth: 0.13,
        rs_pen: 0.8,
        lv_default: "1101, 1105, 1103",
        lv_list: "1101, 1105, 1103",
        p_ev: 10,

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
      formData.append(
        "mv_solar_pv_csv",
        new Blob(["1, 2, 3"], { type: "text/csv" })
      );
      formData.append(
        "mv_ev_charger_csv",
        new Blob(["1, 2, 3"], { type: "text/csv" })
      );
      formData.append(
        "lv_smart_meter_csv",
        new Blob(["1, 2, 3"], { type: "text/csv" })
      );
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
