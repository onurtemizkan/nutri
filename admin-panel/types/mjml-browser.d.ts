declare module 'mjml-browser' {
  interface MJMLParseError {
    line: number;
    message: string;
    tagName: string;
    formattedMessage: string;
  }

  interface MJMLParseResults {
    html: string;
    errors: MJMLParseError[];
  }

  interface MJMLOptions {
    validationLevel?: 'strict' | 'soft' | 'skip';
    minify?: boolean;
    beautify?: boolean;
    filePath?: string;
    keepComments?: boolean;
  }

  function mjml2html(input: string, options?: MJMLOptions): MJMLParseResults;

  export default mjml2html;
}
