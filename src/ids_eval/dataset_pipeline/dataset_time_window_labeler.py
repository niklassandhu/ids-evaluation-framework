import pandas as pd
from datetime import datetime


class TimeWindowLabeler:
    """Label nfstream flows as attack/benign by configured time windows and IPs."""

    def __init__(self, windows):
        self.windows = self._parse_windows(windows)

    def _parse_windows(self, windows):
        self.default_label = "BENIGN"
        parsed = []

        for w in windows:
            #print(w)
            if getattr(w, "default", False):
                self.default_label = w.label
                continue

            parsed.append(w)

        return parsed

    def label(self, df: pd.DataFrame, debug=True, sample_n=5) -> pd.DataFrame:
        """Labeled flows by time window (UTC, matched in both directions) in a Label column"""
        df = df.copy()

        #match corrsponding types NF <-> config
        if pd.api.types.is_datetime64_any_dtype(df["bidirectional_first_seen_ms"]):
            df["flow_start"] = pd.to_datetime(df["bidirectional_first_seen_ms"], utc=True)
        else:
            df["flow_start"] = pd.to_datetime(
                df["bidirectional_first_seen_ms"], unit="ms", utc=True
            )

        df["Label"] = self.default_label

        #correlate the flows according to config with Labels
        for i, w in enumerate(self.windows):
            w_start = pd.Timestamp(w.start)
            w_end = pd.Timestamp(w.end)

            #prepare comparison
            if w_start.tzinfo is None:
                w_start = w_start.tz_localize("UTC")
            else:
                w_start = w_start.tz_convert("UTC")

            if w_end.tzinfo is None:
                w_end = w_end.tz_localize("UTC")
            else:
                w_end = w_end.tz_convert("UTC")

            #DISCLAIMER: this was refactored by Gemini due to my naive, way slower iterative approach
            #vectorize bc of large amounts of data
            # boolean array
            time_mask = (
                (df["flow_start"] >= w_start) &
                (df["flow_start"] <= w_end)
            )
            # 'verunden' with correct IPs
            fwd_mask = time_mask.copy()
            if w.src_ips:
                fwd_mask &= df["src_ip"].isin(w.src_ips)
            if w.dst_ips:
                fwd_mask &= df["dst_ip"].isin(w.dst_ips)

            #bidirectional flows can start both ways
            rev_mask = time_mask.copy()
            if w.dst_ips:
                rev_mask &= df["src_ip"].isin(w.dst_ips)
            if w.src_ips:
                rev_mask &= df["dst_ip"].isin(w.src_ips)

            #'verodern' for bidirectinality
            match_mask = fwd_mask | rev_mask

            df.loc[match_mask, "Label"] = w.label

        # print(df["Label"].value_counts())
        return df








# Label from NFStreamer:
    #    'id', 'expiration_id', 'src_ip', 'src_mac', 'src_oui', 'src_port',
    #    'dst_ip', 'dst_mac', 'dst_oui', 'dst_port', 'protocol', 'ip_version',
    #    'vlan_id', 'tunnel_id', 'bidirectional_first_seen_ms',
    #    'bidirectional_last_seen_ms', 'bidirectional_duration_ms',
    #    'bidirectional_packets', 'bidirectional_bytes', 'src2dst_first_seen_ms',
    #    'src2dst_last_seen_ms', 'src2dst_duration_ms', 'src2dst_packets',
    #    'src2dst_bytes', 'dst2src_first_seen_ms', 'dst2src_last_seen_ms',
    #    'dst2src_duration_ms', 'dst2src_packets', 'dst2src_bytes',
    #    'bidirectional_min_ps', 'bidirectional_mean_ps',
    #    'bidirectional_stddev_ps', 'bidirectional_max_ps', 'src2dst_min_ps',
    #    'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps',
    #    'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
    #    'dst2src_max_ps', 'bidirectional_min_piat_ms',
    #    'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
    #    'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
    #    'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms',
    #    'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms',
    #    'dst2src_max_piat_ms', 'bidirectional_syn_packets',
    #    'bidirectional_cwr_packets', 'bidirectional_ece_packets',
    #    'bidirectional_urg_packets', 'bidirectional_ack_packets',
    #    'bidirectional_psh_packets', 'bidirectional_rst_packets',
    #    'bidirectional_fin_packets', 'src2dst_syn_packets',
    #    'src2dst_cwr_packets', 'src2dst_ece_packets', 'src2dst_urg_packets',
    #    'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets',
    #    'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_cwr_packets',
    #    'dst2src_ece_packets', 'dst2src_urg_packets', 'dst2src_ack_packets',
    #    'dst2src_psh_packets', 'dst2src_rst_packets', 'dst2src_fin_packets',
    #    'application_name', 'application_category_name',
    #    'application_is_guessed', 'application_confidence',
    #    'requested_server_name', 'client_fingerprint', 'server_fingerprint',
    #    'user_agent', 'content_type'],
    #   dtype='str')