export default function CodeOutputBlock(props) {
  const { children } = props;
  // add styling that makes this block a light gray
  const inlineStyling = {
    backgroundColor: "#f5f5f5",
    padding: "1rem",
    borderRadius: "5px",
    marginBottom: "1rem",
  };
  return <div
    className="code-output-block"
    dangerouslySetInnerHTML={props.dangerouslySetInnerHTML}
    style={inlineStyling}
  >{children}</div>;
}